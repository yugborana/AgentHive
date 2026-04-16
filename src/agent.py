"""Agent Orchestrator for AgentHive."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Callable

from .messages import AssistantMessage, Message, SystemPrompt, ToolMessage, UserPrompt
from .messages import AssistantMessage, Message, SystemPrompt, ToolMessage, UserPrompt
from .tools import Tool
from .result import OutputSchema
from .llm import LLM, ToolSchema
from .memory import MemoryStore


@dataclass
class AgentResult:
    """What the agent returns after a successful run.

    Attributes:
        data: The final result — either a string (plain text) or
              a Pydantic model instance (structured output).
        messages: The full conversation history, useful for
                  debugging, logging, or continuing the conversation.
    """
    data: Any
    messages: list[Message] = field(repr=False)


class Agent:
    """The Orchestrator.

    Ties the LLM, tools, and structured output together into
    a single .run() call.
    """

    def __init__(
        self,
        llm: LLM,
        *,
        system_prompt: str = "",
        tools: list[Callable[..., Any]] | None = None,
        result_type: type | None = None,
        max_turns: int = 10,
        memory: MemoryStore | None = None,
    ):
        self._llm = llm
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._memory = memory

        # Wrap raw Python functions into Tool objects
        self._tools: dict[str, Tool] = {}
        if tools:
            for func in tools:
                tool = Tool(func)
                self._tools[tool.name] = tool

        # Build structured output schema (the "fake tool" trick)
        self._output_schema: OutputSchema | None = None
        if result_type is not None:
            self._output_schema = OutputSchema(result_type)

    async def run(
        self,
        user_prompt: str,
        message_history: list[Message] | None = None,
        session_id: str | None = None,
    ) -> AgentResult:
        """Run the agent. This is the entire engine.

        The flow:
            1. Build the initial messages (system + user prompt)
            2. Gather all tool schemas (real tools + fake result tool)
            3. Enter the while True loop:
                a. Send messages to LLM
                b. LLM replies with text or tool calls
                c. If text → return it (unless structured output is expected)
                d. If tool call → execute it and append result to history
                e. If "final_result" tool → validate and return structured data
                f. Loop back to (a) with updated history

        Args:
            user_prompt: The user's question or instruction.
            message_history: Optional prior conversation to continue from.

        Returns:
            AgentResult with the final data and full message history.
        """
        # --- Step 1: Build the conversation ---
        messages = []
        if self._system_prompt:
            messages.append(SystemPrompt(role="system", content=self._system_prompt))
            
        # Optional: Load historic conversation from DB
        if session_id and self._memory:
            db_history = await self._memory.get_messages(session_id)
            messages.extend(db_history)
            
        # Optional: Manually injected conversation
        if message_history is not None:
            messages.extend(message_history)

        # Append the new user prompt
        messages.append(UserPrompt(role="user", content=user_prompt))
        
        # Track where the *new* messages start so we only save those
        start_index = len(messages) - 1

        # --- Step 2: Gather all tool schemas ---
        tool_schemas: list[ToolSchema] = [
            tool.schema for tool in self._tools.values()
        ]
        if self._output_schema:
            tool_schemas.append(self._output_schema.tool_schema)

        # Helper to safely save state before returning
        async def save_state():
            if session_id and self._memory:
                new_messages = [m for m in messages[start_index:] if m["role"] != "system"]
                await self._memory.add_messages(session_id, new_messages)

        # --- Step 3: The Loop ---
        for turn in range(self._max_turns):
            # (a) Send everything to the LLM
            response = await self._llm.chat(
                messages,
                tools=tool_schemas if tool_schemas else None,
            )

            # Always append the LLM's response to history
            messages.append(response)

            # (b) Did the LLM call any tools?
            tool_calls = response.get("tool_calls")

            if not tool_calls:
                # --- Case 1: Plain text response ---
                if self._output_schema:
                    # The LLM was supposed to use the final_result tool
                    # but replied with plain text instead. Nudge it back.
                    messages.append(
                        UserPrompt(
                            role="user",
                            content="Please use the final_result tool to provide your response.",
                        )
                    )
                    continue
                else:
                    # Plain text is exactly what we wanted. Done!
                    await save_state()
                    return AgentResult(
                        data=response.get("content", ""),
                        messages=messages,
                    )

            # --- Case 2: Tool calls ---
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                arguments_json = tc["function"]["arguments"]
                tool_call_id = tc["id"]

                # Is it the fake "final_result" tool? (structured output)
                if self._output_schema and func_name == "final_result":
                    result = self._output_schema.validate(arguments_json)

                    if isinstance(result, str):
                        # Validation failed. Bounce the error back.
                        messages.append(
                            ToolMessage(
                                role="tool",
                                tool_call_id=tool_call_id,
                                content=result,
                            )
                        )
                    else:
                        # Validation succeeded! The loop is over.
                        await save_state()
                        return AgentResult(data=result, messages=messages)

                # Is it a real tool?
                elif func_name in self._tools:
                    try:
                        result_str = await self._tools[func_name].execute(
                            arguments_json
                        )
                    except Exception as e:
                        result_str = f"Tool execution error: {e}"

                    messages.append(
                        ToolMessage(
                            role="tool",
                            tool_call_id=tool_call_id,
                            content=result_str,
                        )
                    )

                # Unknown tool — the LLM hallucinated a tool name
                else:
                    messages.append(
                        ToolMessage(
                            role="tool",
                            tool_call_id=tool_call_id,
                            content=f"Error: Unknown tool '{func_name}'. "
                            f"Available tools: {list(self._tools.keys())}",
                        )
                    )

            await save_state()
            raise RuntimeError(
                f"Agent exceeded {self._max_turns} turns without producing a final result."
            )

    def run_sync(self, user_prompt: str) -> AgentResult:
        """Convenience wrapper to run the agent synchronously."""
        return asyncio.run(self.run(user_prompt))

    def as_tool(self, name: str, description: str) -> Callable:
        """Wrap this agent into a standalone tool that another agent can call.
        
        This implements the "Agent Delegation" Multi-Agent pattern. 
        When the parent agent calls this tool, it suspends, and this
        child agent takes over in a fresh loop until it reaches a final answer,
        which is then passed back to the parent.
        """
        async def delegate_agent_tool(instruction_for_agent: str) -> str:
            # We run a brand new autonomous loop for the child agent!
            # We don't pass session_id because we want to keep the 
            # child's thought process isolated from the parent's memory.
            result = await self.run(instruction_for_agent)
            
            # If the child agent returned a structured Pydantic object,
            # we cast it to a string so the parent LLM can read it.
            return str(result.data)

        # We must overwrite the __name__ and __doc__ so our Tool Introspection
        # engine creates the correct JSON schema for the parent LLM to see!
        delegate_agent_tool.__name__ = name
        delegate_agent_tool.__doc__ = (
            f"{description}\n\n"
            "Args:\n"
            "    instruction_for_agent: The prompt or task to give to the sub-agent."
        )
        
        return delegate_agent_tool

    def run_stream(
        self,
        user_prompt: str,
        message_history: list[Message] | None = None,
        session_id: str | None = None,
    ) -> AgentStream:
        """Run the agent and stream the output.
        
        Usage:
            stream = agent.run_stream("Tell me a story", session_id="user123")
            async for chunk in stream.stream_text():
                print(chunk, end="")
            result = await stream.get_data()
        """
        return AgentStream(self, user_prompt, message_history, session_id)


class AgentStream:
    """Wraps an agent execution to provide streaming text and a final result."""
    
    def __init__(self, agent: Agent, user_prompt: str, message_history: list[Message] | None, session_id: str | None):
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run_loop(agent, user_prompt, message_history, session_id))

    async def stream_text(self) -> AsyncIterator[str]:
        """Iterate over text chunks as they arrive from the LLM."""
        while True:
            chunk = await self._queue.get()
            if chunk is None:  # Sentinel value indicating the stream is done
                break
            yield chunk

    async def get_data(self) -> AgentResult:
        """Wait for the agent to finish and get the final result."""
        return await self._task

    async def _run_loop(self, agent: Agent, user_prompt: str, message_history: list[Message] | None, session_id: str | None) -> AgentResult:
        """The internal loop, adapted for streaming."""
        
        # --- Step 1: Build the conversation ---
        messages = []
        if agent._system_prompt:
            messages.append(SystemPrompt(role="system", content=agent._system_prompt))
            
        if session_id and agent._memory:
            db_history = await agent._memory.get_messages(session_id)
            messages.extend(db_history)

        if message_history is not None:
            messages.extend(message_history)

        messages.append(UserPrompt(role="user", content=user_prompt))
        start_index = len(messages) - 1

        # --- Step 2: Gather all tool schemas ---
        tool_schemas: list[ToolSchema] = [
            tool.schema for tool in agent._tools.values()
        ]
        if agent._output_schema:
            tool_schemas.append(agent._output_schema.tool_schema)
            
        async def save_state():
            if session_id and agent._memory:
                new_messages = [m for m in messages[start_index:] if m["role"] != "system"]
                await agent._memory.add_messages(session_id, new_messages)

        # --- Step 3: The Loop ---
        try:
            for turn in range(agent._max_turns):
                # (a) Send everything to the LLM (STREAMING)
                stream_response = await agent._llm.chat_stream(
                    messages,
                    tools=tool_schemas if tool_schemas else None,
                )

                # Yield all incoming text chunks to the user
                async for chunk in stream_response:
                    await self._queue.put(chunk)

                # After text stream ends, get the full assembled message
                response = stream_response.get_final_message()
                messages.append(response)

                # (b) Did the LLM call any tools?
                tool_calls = response.get("tool_calls")

                if not tool_calls:
                    # --- Case 1: Plain text response ---
                    if agent._output_schema:
                        messages.append(
                            UserPrompt(
                                role="user",
                                content="Please use the final_result tool to provide your response.",
                            )
                        )
                        continue
                    else:
                        await save_state()
                        return AgentResult(data=response.get("content", ""), messages=messages)

                # --- Case 2: Tool calls ---
                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    arguments_json = tc["function"]["arguments"]
                    tool_call_id = tc["id"]

                    if agent._output_schema and func_name == "final_result":
                        result = agent._output_schema.validate(arguments_json)

                        if isinstance(result, str):
                            messages.append(
                                ToolMessage(
                                    role="tool",
                                    tool_call_id=tool_call_id,
                                    content=result,
                                )
                            )
                        else:
                            await save_state()
                            return AgentResult(data=result, messages=messages)

                    elif func_name in agent._tools:
                        try:
                            result_str = await agent._tools[func_name].execute(arguments_json)
                        except Exception as e:
                            result_str = f"Tool execution error: {e}"

                        messages.append(
                            ToolMessage(
                                role="tool",
                                tool_call_id=tool_call_id,
                                content=result_str,
                            )
                        )

                    else:
                        messages.append(
                            ToolMessage(
                                role="tool",
                                tool_call_id=tool_call_id,
                                content=f"Error: Unknown tool '{func_name}'. "
                                f"Available tools: {list(agent._tools.keys())}",
                            )
                        )

            await save_state()
            raise RuntimeError(f"Agent exceeded {agent._max_turns} turns without producing a final result.")
            
        finally:
            # Always push the sentinel so the stream_text iterator finishes
            await self._queue.put(None)
