"""Live integration test: Python Function → Tool Introspection → Groq LLM → Execution.

Run from the project root:
    python -m tests.test_agent_tools_live
"""

import os
import sys
import json
import asyncio

# Hack to make the 'src' module visible when running the file directly from the IDE
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.messages import SystemPrompt, UserPrompt
from src.llm.groq import GroqLLM
from src.tools import Tool

API_KEY = os.environ.get("GROQ_API_KEY", "your_api_key_here")


# ---------------------------------------------------------------------------
# 1. Provide an ordinary Python function
# ---------------------------------------------------------------------------
def get_flight_price(origin: str, destination: str, class_type: str = "economy"):
    """Fetch the real-time flight price between two cities.
    
    Args:
        origin: Airport code to depart from (e.g. "JFK")
        destination: Airport code to arrive at (e.g. "LHR")
        class_type: The seat class, 'economy' or 'business'
    """
    print(f"\n[SERVER] -> Executing database lookup for flight {origin} to {destination} in {class_type}...")
    
    # Fake database logic
    base_price = 450
    if class_type.lower() == "business":
        base_price *= 3
        
    return {"price_usd": base_price, "available_seats": 12}


async def main():
    print("=" * 60)
    print("INTEGRATION TEST: Automatic Tool Introspection + LLM")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 2. Introspect it into an AgentHive Tool (Magic happens here)
    # -----------------------------------------------------------------------
    flight_tool = Tool(get_flight_price)
    
    print("Generated JSON Schema sent to Groq:")
    print(json.dumps(flight_tool.schema.parameters, indent=2))
    print("-" * 60)

    # -----------------------------------------------------------------------
    # 3. Setup LLM and messages
    # -----------------------------------------------------------------------
    llm = GroqLLM("llama-3.3-70b-versatile", api_key=API_KEY)

    messages = [
        SystemPrompt(role="system", content="You are a helpful travel assistant."),
        UserPrompt(role="user", content="How much is a business class ticket from JFK to LHR?"),
    ]

    print(f"User: {messages[1]['content']}")
    
    # Send the request with the automatically generated schema!
    response = await llm.chat(messages, tools=[flight_tool.schema])

    # -----------------------------------------------------------------------
    # 4. Handle the LLM's response
    # -----------------------------------------------------------------------
    if response.get("tool_calls"):
        for tc in response["tool_calls"]:
            func_name = tc["function"]["name"]
            arguments_json = tc["function"]["arguments"]
            
            print(f"LLM decided to call : '{func_name}'")
            print(f"LLM provided args   : {arguments_json}")
            
            # Execute the function using the Tool wrapper
            # This safely validates the arguments and runs `get_flight_price`
            result_str = await flight_tool.execute(arguments_json)
            
            print(f"[SERVER] -> Tool returned: {result_str}")
            
    else:
        print(f"Assistant: {response.get('content')}")

if __name__ == "__main__":
    asyncio.run(main())
