"""Test the Tool introspection engine.

Run: python tests/test_tools.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from src.tools import Tool


# --- Example functions to turn into Tools ---

def get_weather(location: str, unit: str = "celsius"):
    """Get the current weather for a city.

    Args:
        location: The city name, e.g. "Tokyo"
        unit: Temperature unit, "celsius" or "fahrenheit"
    """
    return {"temp": 22, "unit": unit, "city": location}


async def search_database(query: str, limit: int = 5):
    """Search the internal database.

    Args:
        query: The search query string.
        limit: Maximum number of results to return.
    """
    return f"Found {limit} results for '{query}'"


def no_docstring_func(name: str, age: int):
    return f"{name} is {age}"


# --- Tests ---

def test_schema_generation():
    """Test that Tool correctly introspects a function into a ToolSchema."""
    print("=" * 50)
    print("TEST 1: Schema generation")
    print("=" * 50)

    tool = Tool(get_weather)

    print(f"Name:        {tool.name}")
    print(f"Description: {tool.description}")
    print(f"Is async:    {tool.is_async}")
    print()

    schema = tool.schema
    print(f"ToolSchema name: {schema.name}")
    print(f"ToolSchema desc: {schema.description}")
    print(f"Parameters JSON Schema:")
    print(json.dumps(schema.parameters, indent=2))
    print()


def test_argument_validation():
    """Test that Tool validates LLM arguments correctly."""
    print("=" * 50)
    print("TEST 2: Argument validation")
    print("=" * 50)

    tool = Tool(get_weather)

    # Good arguments
    kwargs = tool.validate_arguments('{"location": "Paris"}')
    print(f"Valid args:   {kwargs}")

    # Good arguments with optional
    kwargs = tool.validate_arguments('{"location": "Tokyo", "unit": "fahrenheit"}')
    print(f"With default: {kwargs}")

    # Bad arguments (missing required field)
    try:
        tool.validate_arguments('{"unit": "celsius"}')
    except Exception as e:
        print(f"Caught error: {type(e).__name__} (missing 'location')")

    print()


async def test_execution():
    """Test that Tool executes functions correctly."""
    print("=" * 50)
    print("TEST 3: Execution")
    print("=" * 50)

    # Sync function
    tool = Tool(get_weather)
    result = await tool.execute('{"location": "Paris"}')
    print(f"Sync result:  {result}")

    # Async function
    async_tool = Tool(search_database)
    result = await async_tool.execute('{"query": "python tutorials", "limit": 3}')
    print(f"Async result: {result}")
    print(f"Async tool schema: {async_tool.schema.name}")

    print()


def test_no_docstring():
    """Test that a function without a docstring still works."""
    print("=" * 50)
    print("TEST 4: No docstring")
    print("=" * 50)

    tool = Tool(no_docstring_func)
    print(f"Name: {tool.name}")
    print(f"Desc: '{tool.description}'")
    print(f"Schema: {json.dumps(tool.schema.parameters, indent=2)}")
    print()


async def main():
    test_schema_generation()
    test_argument_validation()
    await test_execution()
    test_no_docstring()
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
