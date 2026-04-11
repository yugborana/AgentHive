"""Groq LLM provider for AgentHive.

Groq's API is OpenAI-compatible. Our message TypedDicts already match
the API format, so conversion is minimal — we just reshape the tool
schemas into Groq's expected structure.
"""

from __future__ import annotations

from groq import AsyncGroq

from ..messages import AssistantMessage, FunctionCall, Message, ToolCall
from . import LLM, ToolSchema


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
