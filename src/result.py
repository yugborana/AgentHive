"""Structured output for AgentHive.

Forces the LLM to reply in a strict JSON format by disguising
the desired output schema as a "fake tool" called 'final_result'.

The trick:
    1. Developer provides a Pydantic model (e.g. WeatherReport)
    2. We turn it into a ToolSchema named 'final_result'
    3. We tell the LLM: tool_choice='required' (you MUST call a tool)
    4. The LLM fills in the arguments → those arguments ARE the structured data
    5. We validate with Pydantic → success = end the loop, failure = bounce error back
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, TypeAdapter, ValidationError

from .llm import ToolSchema

# Generic type variable so OutputSchema can work with ANY Pydantic model
T = TypeVar("T")


@dataclass
class OutputSchema(Generic[T]):
    """Wraps a Pydantic model into a fake tool for structured LLM output.

    Usage:
        class WeatherReport(BaseModel):
            temp: int
            condition: str

        schema = OutputSchema(WeatherReport)
        schema.tool_schema      # → ToolSchema to send to LLM
        schema.validate(json)   # → WeatherReport or error string
    """

    response_type: type[T]
    _type_adapter: TypeAdapter[Any]
    _json_schema: dict

    def __init__(self, response_type: type[T]):
        self.response_type = response_type
        self._type_adapter = TypeAdapter(response_type)
        self._json_schema = self._type_adapter.json_schema()

    @property
    def tool_schema(self) -> ToolSchema:
        """Generate the fake tool that the LLM will be forced to call.

        The LLM sees this as just another tool. But when the agent loop
        detects the LLM called 'final_result', it knows the loop is over.
        """
        schema = dict(self._json_schema)
        schema.pop("title", None)

        return ToolSchema(
            name="final_result",
            description="Return the final structured response. You MUST call this tool with the required fields.",
            parameters=schema,
        )

    def validate(self, arguments_json: str) -> T | str:
        """Validate the LLM's arguments against the Pydantic model.

        This is our simplified version of Pydantic AI's "Either" pattern:
            - Success → returns the parsed Pydantic model instance (T)
            - Failure → returns an error string to bounce back to the LLM

        The agent loop checks: if I got a string back, something went wrong,
        append it as a ToolMessage and let the LLM try again.
        If I got an object back, we're done!

        Args:
            arguments_json: The raw JSON string from the LLM's tool call.

        Returns:
            Either the validated Pydantic model instance, or an error string.
        """
        try:
            return self._type_adapter.validate_json(arguments_json)
        except ValidationError as e:
            # Format the Pydantic errors into a readable string
            # that the LLM can understand and fix
            error_details = []
            for error in e.errors():
                field = " → ".join(str(loc) for loc in error["loc"])
                error_details.append(f"  - {field}: {error['msg']}")

            return (
                "Validation failed. Fix these errors and call final_result again:\n"
                + "\n".join(error_details)
            )
