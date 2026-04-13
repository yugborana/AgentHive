"""Agent Orchestrator for AgentHive."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from .messages import AssistantMessage, Message, SystemPrompt, ToolMessage, UserPrompt
from .tools import Tool
from .result import OutputSchema
from .llm import LLM, ToolSchema


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
    ):
        self._llm = llm
        self._system_prompt = system_prompt
        self._max_turns = max_turns

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
        if message_history is not None:
            messages: list[Message] = message_history.copy()
        else:
            messages = []
            if self._system_prompt:
                messages.append(
                    SystemPrompt(role="system", content=self._system_prompt)
                )

        messages.append(UserPrompt(role="user", content=user_prompt))

        # --- Step 2: Gather all tool schemas ---
        tool_schemas: list[ToolSchema] = [
            tool.schema for tool in self._tools.values()
        ]
        if self._output_schema:
            tool_schemas.append(self._output_schema.tool_schema)

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
                        # Don't return — the loop continues, LLM tries again
                    else:
                        # Validation succeeded! The loop is over.
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

        # If we exhaust max_turns, return whatever we have
        raise RuntimeError(
            f"Agent exceeded {self._max_turns} turns without producing a final result."
        )

    def run_sync(self, user_prompt: str) -> AgentResult:
        """Convenience wrapper to run the agent synchronously.

        Usage:
            result = agent.run_sync("What is Python?")
        """
        return asyncio.run(self.run(user_prompt))
