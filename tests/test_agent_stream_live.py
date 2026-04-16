"""Live test for the Streaming Agent Orchestrator.

Run: python tests/test_agent_stream_live.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from pydantic import BaseModel
from src.agent import Agent
from src.llm.groq import GroqLLM

API_KEY = os.environ.get("GROQ_API_KEY", "your_api_key_here")
llm = GroqLLM("llama-3.3-70b-versatile", api_key=API_KEY)


async def test_text_streaming():
    print("=" * 60)
    print("TEST 1: Text Streaming")
    print("=" * 60)

    agent = Agent(llm, system_prompt="You are a poet. Reply in exactly 4 sentences.")
    
    stream = agent.run_stream("Tell me a story about a fast computer.")
    
    print("Response Stream: ", end="", flush=True)
    async for chunk in stream.stream_text():
        print(chunk, end="", flush=True)
        # Small delay just to make the streaming visual effect more obvious in tests
        await asyncio.sleep(0.02) 
    print("\n\n")

    result = await stream.get_data()
    print("Stream finished! Total messages in history:", len(result.messages))
    print()


def get_weather(location: str):
    """Get the current weather for a city.
    
    Args:
        location: The city name.
    """
    return f"The weather in {location} is beautiful and 25°C."


async def test_tool_streaming():
    print("=" * 60)
    print("TEST 2: Tool Calling while Streaming")
    print("=" * 60)

    agent = Agent(llm, tools=[get_weather], system_prompt="You are a weather assistant.")
    
    # 1. First it streams nothing (tool call silently runs)
    # 2. Then the agent loop sends tool result back to LLM
    # 3. LLM streams the final response
    stream = agent.run_stream("What's the weather in Seattle?")
    
    print("Response Stream: ", end="", flush=True)
    async for chunk in stream.stream_text():
        print(chunk, end="", flush=True)
    print("\n\n")

    result = await stream.get_data()
    print("Interaction complete. Messages involved:")
    for msg in result.messages:
        print(f" - {msg['role']}")
    print()


if __name__ == "__main__":
    asyncio.run(test_text_streaming())
    asyncio.run(test_tool_streaming())
