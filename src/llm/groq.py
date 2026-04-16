"""Groq LLM provider for AgentHive.

Groq's API is OpenAI-compatible. Our message TypedDicts already match
the API format, so conversion is minimal — we just reshape the tool
schemas into Groq's expected structure.
"""

from __future__ import annotations

from groq import AsyncGroq

from collections.abc import AsyncIterator

from ..messages import AssistantMessage, FunctionCall, Message, ToolCall
from . import LLM, StreamResponse, ToolSchema


class GroqLLM(LLM):
    """Groq provider.

    Usage:
        llm = GroqLLM("llama-3.3-70b-versatile")
        response = await llm.chat(messages)
    """

    def __init__(self, model: str, *, api_key: str | None = None):
        self.model = model
        # If api_key is None, the SDK reads GROQ_API_KEY from env
        self._client = AsyncGroq(api_key=api_key)

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> AssistantMessage:

        # Build the API call kwargs
        kwargs: dict = {
            "model": self.model,
            "messages": messages,  # Our TypedDicts already match the API format
        }

        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        # Convert the SDK response → our AssistantMessage
        if choice.message.tool_calls:
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(
                        id=tc.id,
                        type="function",
                        function=FunctionCall(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                    for tc in choice.message.tool_calls
                ],
            )

        return AssistantMessage(
            role="assistant",
            content=choice.message.content or "",
        )

    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> StreamResponse:
        
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            kwargs["tool_choice"] = "auto"

        stream_request = await self._client.chat.completions.create(**kwargs)
        return GroqStreamResponse(stream_request)


class GroqStreamResponse(StreamResponse):
    """Handles the Groq/OpenAI streaming response format.
    
    Yields text chunks immediately.
    Silently accumulates tool call chunks in the background.
    """

    def __init__(self, stream_request):
        self._stream_request = stream_request
        self._content = ""
        self._tool_calls: dict[int, dict] = {}
        self._is_resolved = False

    async def __aiter__(self) -> AsyncIterator[str]:
        async for chunk in self._stream_request:
            if not chunk.choices:
                continue
                
            delta = chunk.choices[0].delta
            
            # 1. Text chunks (Yield these)
            if delta.content:
                self._content += delta.content
                yield delta.content
                
            # 2. Tool call chunks (Accumulate these silently)
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in self._tool_calls:
                        self._tool_calls[idx] = {"id": tc.id, "name": "", "arguments": ""}
                    
                    if tc.function:
                        if tc.function.name:
                            self._tool_calls[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            self._tool_calls[idx]["arguments"] += tc.function.arguments

        self._is_resolved = True

    def get_final_message(self) -> AssistantMessage:
        if not self._is_resolved:
            raise RuntimeError("You must finish iterating over the stream before getting the final message.")
            
        if self._tool_calls:
            # We got a tool call (like structured output), build the message!
            formatted_calls = [
                ToolCall(
                    id=call["id"],
                    type="function",
                    function=FunctionCall(
                        name=call["name"],
                        arguments=call["arguments"],
                    ),
                )
                for call in self._tool_calls.values()
            ]
            
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=formatted_calls,
            )

        # We got a regular chat message
        return AssistantMessage(
            role="assistant",
            content=self._content,
        )
