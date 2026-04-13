"""Live test for the Agent Orchestrator.

Tests all three modes:
    1. Plain text (no tools, no structured output)
    2. Tool calling (agent uses a real python function)
    3. Structured output (agent returns a validated Pydantic model)

Run: python tests/test_agent_live.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from pydantic import BaseModel

from src.agent import Agent
from src.llm.groq import GroqLLM


# --- Setup ---
llm = GroqLLM("llama-3.3-70b-versatile", api_key=os.environ.get("GROQ_API_KEY", "your_api_key_here"))


# =====================================================================
# TEST 1: Plain text — just chat, no tools, no structured output
# =====================================================================
async def test_plain_text():
    print("=" * 60)
    print("TEST 1: Plain text chat")
    print("=" * 60)

    agent = Agent(llm, system_prompt="You are a helpful assistant. Keep answers short.")

    result = await agent.run("What is Python in one sentence?")

    print(f"Response: {result.data}")
    print(f"Messages exchanged: {len(result.messages)}")
    print()


# =====================================================================
# TEST 2: Tool calling — agent autonomously calls a Python function
# =====================================================================
def get_weather(location: str, unit: str = "celsius"):
    """Get the current weather for a city.

    Args:
        location: The city name, e.g. "Tokyo"
        unit: Temperature unit, "celsius" or "fahrenheit"
    """
    # Fake database
    data = {"Tokyo": 28, "Paris": 18, "London": 14}
    temp = data.get(location, 22)
    return f"{temp}°{'C' if unit == 'celsius' else 'F'}, partly cloudy"


async def test_tool_calling():
    print("=" * 60)
    print("TEST 2: Tool calling")
    print("=" * 60)

    agent = Agent(
        llm,
        system_prompt="You are a weather assistant. Use the get_weather tool to answer.",
        tools=[get_weather],
    )

    result = await agent.run("What's the weather like in Tokyo?")

    print(f"Response: {result.data}")
    print(f"Messages exchanged: {len(result.messages)}")
    print()


# =====================================================================
# TEST 3: Structured output — agent returns a validated Pydantic model
# =====================================================================
class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str
    recommended: bool


async def test_structured_output():
    print("=" * 60)
    print("TEST 3: Structured output")
    print("=" * 60)

    agent = Agent(
        llm,
        system_prompt="You are a professional movie critic. Always use the final_result tool.",
        result_type=MovieReview,
    )

    result = await agent.run("Review the movie 'The Dark Knight'")

    print(f"Type:        {type(result.data).__name__}")
    print(f"Title:       {result.data.title}")
    print(f"Rating:      {result.data.rating}")
    print(f"Summary:     {result.data.summary}")
    print(f"Recommended: {result.data.recommended}")
    print(f"Messages exchanged: {len(result.messages)}")
    print()


# =====================================================================
# Run all tests
# =====================================================================
async def main():
    await test_plain_text()
    await test_tool_calling()
    await test_structured_output()
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
