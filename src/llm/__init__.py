"""Core LLM abstraction for AgentHive."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

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