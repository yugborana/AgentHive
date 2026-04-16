"""Live test for the Agent Memory / Checkpointing.

Demonstrates how an Agent remembers context across multiple `.run()` 
calls by utilizing the InMemoryStore.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from src.agent import Agent
from src.llm.groq import GroqLLM
from src.memory import InMemoryStore

API_KEY = os.environ.get("GROQ_API_KEY", "your_api_key_here")
llm = GroqLLM("llama-3.3-70b-versatile", api_key=API_KEY)


async def main():
    print("=" * 60)
    print("TEST: Agent with Persistent Memory")
    print("=" * 60)

    # 1. Initialize the memory store
    # (We use InMemoryStore for testing, but you could use SQLiteMemoryStore
    #  to persist across script restarts!)
    memory_db = InMemoryStore()

    agent = Agent(
        llm, 
        system_prompt="You are a helpful assistant. Keep your answers to one sentence.",
        memory=memory_db
    )
    
    session_id = "user_johndoe_123"

    # --- Turn 1 ---
    print("\n[Turn 1] User: Hi, my name is Alex and my favorite color is Blue.")
    result1 = await agent.run(
        "Hi, my name is Alex and my favorite color is Blue.",
        session_id=session_id
    )
    print(f"Assistant: {result1.data}")
    
    # --- Turn 2 ---
    print("\n[Turn 2] User: What is my name and favorite color? Do you remember?")
    
    # Notice we don't pass `message_history`! 
    # The agent automatically pulls it from memory_db using the session_id!
    result2 = await agent.run(
        "What is my name and favorite color? Do you remember?",
        session_id=session_id
    )
    print(f"Assistant: {result2.data}")

    # Let's inspect the database directly to see what it saved
    saved_history = await memory_db.get_messages(session_id)
    print(f"\n[Database Check] The database has saved {len(saved_history)} messages for {session_id}.")
    print("Notice the System Prompt is NOT saved to the DB, allowing you to change AI personality easily later.")


if __name__ == "__main__":
    asyncio.run(main())
