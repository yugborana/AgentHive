"""Core LLM abstraction for AgentHive."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from collections.abc import AsyncIterator

from ..messages import AssistantMessage, Message


@dataclass
class ToolSchema:
    """What a tool looks like to the LLM.

    This is a plain dataclass — not a Protocol, not an ABC.
    The Tool introspection engine (built later) will produce these
    automatically from Python function signatures.

    Example:
        schema = ToolSchema(
            name="get_weather",
            description="Get the current weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        )
    """
    name: str
    description: str
    parameters: dict  # JSON Schema object


class LLM(ABC):
    """Base class for all LLM providers.

    Subclasses must implement one method: chat().
    Everything else (client setup, API format conversion)
    lives inside the subclass — not in shared base code.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> AssistantMessage:
        """Send a conversation to the LLM and get a response.

        Args:
            messages: Full conversation history.
            tools: Tool schemas the LLM can choose to call. None = no tools.

        Returns:
            An AssistantMessage with either:
              - content (the LLM is talking)
              - tool_calls (the LLM wants to run a function)
        """
        raise NotImplementedError(f"{type(self).__name__} must implement chat()")

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> StreamResponse:
        """Send a conversation to the LLM and get a streaming response.

        Returns:
            A StreamResponse object that can be iterated for text chunks,
            and provides the complete AssistantMessage when finished.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement chat_stream()")

class StreamResponse(ABC):
    """A streaming response from an LLM.

    Usage:
        stream = await llm.chat_stream(messages)
        async for chunk in stream:
            print(chunk, end="", flush=True)
            
        final_message = stream.get_final_message()
    """

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[str]:
        """Iterate over the text chunks as they arrive."""
        ...

    @abstractmethod
    def get_final_message(self) -> AssistantMessage:
        """Get the complete message after streaming is finished.
        
        This will contain the full text, OR the assembled tool calls
        if the LLM decided to call tools instead of speaking.
        """
        ...