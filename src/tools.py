"""Tool system for AgentHive.

This module turns plain Python functions into tools the LLM can call.

The pipeline:
    1. Developer writes a normal Python function with type hints + docstring
    2. Tool() introspects it: reads the signature, parses the docstring,
       builds a JSON schema via Pydantic's create_model
    3. The Agent sends the ToolSchema to the LLM
    4. The LLM calls the tool → Agent validates args → executes the function
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from pydantic import ValidationError, create_model
from pydantic.fields import FieldInfo

from .llm import ToolSchema


@dataclass
class Tool:
    """A wrapped Python function that the LLM can call.

    Usage:
        def get_weather(location: str, unit: str = "celsius"):
            '''Get the current weather for a city.

            Args:
                location: The city name, e.g. "Tokyo"
                unit: Temperature unit, "celsius" or "fahrenheit"
            '''
            return f"Weather in {location}: 22°{unit[0].upper()}"

        tool = Tool(get_weather)
        tool.schema      # → ToolSchema ready for the LLM
        tool.execute(...) # → runs the function with validated args
    """

    name: str
    description: str
    function: Callable[..., Any]
    is_async: bool
    _pydantic_model: Any = field(repr=False)  # The dynamically built Pydantic model
    _param_descriptions: dict[str, str] = field(repr=False, default_factory=dict)

    def __init__(self, func: Callable[..., Any]):
        """Build a Tool by introspecting a Python function.

        This is where all the magic happens:
        1. Read the function's name, docstring, type hints
        2. Parse the docstring for parameter descriptions
        3. Build a dynamic Pydantic model from the parameters
        """
        self.name = func.__name__
        self.function = func
        self.is_async = inspect.iscoroutinefunction(func)

        # Step 1: Parse the docstring
        self.description, self._param_descriptions = _parse_docstring(func)

        # Step 2: Read type hints (skip 'return' hint)
        hints = get_type_hints(func)
        hints.pop("return", None)

        # Step 3: Read the signature to find defaults
        sig = inspect.signature(func)

        # Step 4: Build Pydantic model fields from the parameters
        #   Each field = (type, FieldInfo with description and default)
        model_fields: dict[str, Any] = {}
        for param_name, param in sig.parameters.items():
            annotation = hints.get(param_name, Any)
            description = self._param_descriptions.get(param_name, None)

            if param.default is not inspect.Parameter.empty:
                # Has a default value → optional field
                model_fields[param_name] = (
                    annotation,
                    FieldInfo(default=param.default, description=description),
                )
            else:
                # No default → required field
                model_fields[param_name] = (
                    annotation,
                    FieldInfo(description=description),
                )

        # Step 5: Dynamically create a Pydantic model
        #   This is equivalent to writing:
        #     class GetWeather(BaseModel):
        #         location: str = Field(description="The city name")
        #         unit: str = Field(default="celsius", description="...")
        self._pydantic_model = create_model(
            func.__name__,
            **model_fields,
        )

    @property
    def schema(self) -> ToolSchema:
        """Generate the ToolSchema that gets sent to the LLM."""
        json_schema = self._pydantic_model.model_json_schema()

        # Remove the 'title' key — the LLM doesn't need it,
        # and it clutters the schema
        json_schema.pop("title", None)

        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=json_schema,
        )

    def validate_arguments(self, arguments_json: str) -> dict[str, Any]:
        """Validate and parse the LLM's JSON arguments string.

        Args:
            arguments_json: Raw JSON string from the LLM's tool call,
                e.g. '{"location": "Paris"}'

        Returns:
            A validated Python dict of kwargs, ready to pass to the function.

        Raises:
            ValidationError: If the LLM provided bad/missing arguments.
        """
        parsed = self._pydantic_model.model_validate_json(arguments_json)
        return parsed.model_dump()

    async def execute(self, arguments_json: str) -> str:
        """Validate the arguments and run the function.

        Args:
            arguments_json: Raw JSON string from the LLM's tool call.

        Returns:
            The function's return value, converted to a string.
        """
        kwargs = self.validate_arguments(arguments_json)

        if self.is_async:
            result = await self.function(**kwargs)
        else:
            result = self.function(**kwargs)

        # Convert the result to a string for the ToolMessage content
        if isinstance(result, str):
            return result
        return json.dumps(result)


# ---------------------------------------------------------------------------
# Docstring parser (simplified, no external dependencies)
# ---------------------------------------------------------------------------

def _parse_docstring(func: Callable[..., Any]) -> tuple[str, dict[str, str]]:
    """Extract function description and parameter descriptions from a docstring.

    Supports Google-style docstrings:
        '''Main description of the function.

        Args:
            param_name: Description of the parameter.
            other_param: Description of another parameter.
        '''

    Returns:
        (function_description, {param_name: param_description})
    """
    doc = func.__doc__
    if not doc:
        return "", {}

    lines = doc.strip().splitlines()

    # Everything before "Args:" is the main description
    description_lines: list[str] = []
    param_descriptions: dict[str, str] = {}
    in_args_section = False

    for line in lines:
        stripped = line.strip()

        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_args_section = True
            continue

        if stripped.lower().startswith(("returns:", "raises:", "yields:", "examples:", "example:", "note:", "notes:")):
            in_args_section = False
            continue

        if in_args_section and ":" in stripped:
            # Parse "param_name: description" or "param_name (type): description"
            param_part, _, desc_part = stripped.partition(":")
            param_name = param_part.strip().split("(")[0].strip()
            if param_name:
                param_descriptions[param_name] = desc_part.strip()
        elif not in_args_section:
            description_lines.append(stripped)

    description = " ".join(line for line in description_lines if line).strip()
    return description, param_descriptions
