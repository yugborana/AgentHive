"""Live test for the Multi-Agent Delegation pattern.

Demonstrates how a "Manager" agent can delegate complex tasks to
a specialized "Researcher" agent using the `.as_tool()` method.
"""

import os
import sys
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import Agent
from src.llm.groq import GroqLLM

API_KEY = os.environ.get("GROQ_API_KEY", "your_api_key_here")
llm = GroqLLM("llama-3.3-70b-versatile", api_key=API_KEY)


def get_current_stock_price(symbol: str) -> str:
    """Gets the current stock price for a symbol."""
    prices = {"AAPL": "$150", "NVDA": "$900", "TSLA": "$200"}
    return prices.get(symbol.upper(), "Stock not found")


async def main():
    print("=" * 60)
    print("TEST: Multi-Agent Delegation (Manager -> Researcher)")
    print("=" * 60)

    # --- Step 1: Create the Child Agent (The Delegate) ---
    # This agent is given specific tools and hyper-focused instructions.
    research_agent = Agent(
        llm,
        system_prompt="You are a financial researcher. Given a company, find its stock symbol and use get_current_stock_price to find the price. Return ONLY the price.",
        tools=[get_current_stock_price],
    )

    # --- Step 2: Convert the Child Agent into a Tool ---
    # We turn the entire inner Agent into a standard Python function
    # that the parent LLM can call like any other tool.
    research_tool = research_agent.as_tool(
        name="research_stock_price",
        description="Call this agent to research the stock price of any company by providing the company name as an instruction.",
    )

    # --- Step 3: Create the Parent Agent (The Manager) ---
    # This agent has NO tools except the ability to ask the research agent!
    manager_agent = Agent(
        llm,
        system_prompt="You are a friendly hedge fund manager. Use the research_stock_price tool to answer questions, then write a nice 1-sentence reply to the client.",
        tools=[research_tool],
    )

    print("Client: Should I buy Apple stock? I don't know the current price.")
    
    # We use run_stream so we can watch the Manager agent's final text in real-time
    print("Manager: ", end="", flush=True)
    
    stream = manager_agent.run_stream("Should I buy Apple stock? I don't know the current price.")
    async for chunk in stream.stream_text():
        print(chunk, end="", flush=True)
        await asyncio.sleep(0.02)
        
    result = await stream.get_data()
    
    print("\n\n[Behind the scenes checking message history]")
    print(f"Total steps taken by Manager: {len(result.messages)}")


if __name__ == "__main__":
    asyncio.run(main())
