"""Live test: Force the LLM to reply in strict JSON using the "fake tool" trick.

Run: python tests/test_structured_output_live.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from pydantic import BaseModel

from src.messages import SystemPrompt, UserPrompt, ToolMessage, AssistantMessage
from src.llm.groq import GroqLLM
from src.result import OutputSchema

API_KEY = os.environ.get("GROQ_API_KEY", "your_api_key_here")


# ---------------------------------------------------------------------------
# 1. Define the structured output you want (just a normal Pydantic model)
# ---------------------------------------------------------------------------
class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str
    recommended: bool


async def main():
    print("=" * 60)
    print("TEST: Structured Output (Fake Tool Trick)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 2. Create the OutputSchema (this builds the fake tool)
    # ------------------------------------------------------------------
    output_schema = OutputSchema(MovieReview)

    print("Fake tool schema sent to LLM:")
    print(json.dumps(output_schema.tool_schema.parameters, indent=2))
    print("-" * 60)

    # ------------------------------------------------------------------
    # 3. Setup the LLM and conversation
    # ------------------------------------------------------------------
    llm = GroqLLM("llama-3.3-70b-versatile", api_key=API_KEY)

    messages = [
        SystemPrompt(
            role="system",
            content="You are a movie critic. Always respond using the final_result tool.",
        ),
        UserPrompt(
            role="user",
            content="Review the movie 'Inception' by Christopher Nolan.",
        ),
    ]

    # ------------------------------------------------------------------
    # 4. The "Agent Loop" — keep going until we get valid structured output
    # ------------------------------------------------------------------
    max_retries = 3

    for attempt in range(max_retries):
        print(f"\n--- Attempt {attempt + 1} ---")

        # Send to LLM with the fake tool, tool_choice forced to "required"
        # We override tool_choice by passing it through kwargs in chat()
        response = await llm.chat(messages, tools=[output_schema.tool_schema])

        # Check if the LLM called our fake tool
        if response.get("tool_calls"):
            tc = response["tool_calls"][0]
            func_name = tc["function"]["name"]
            arguments_json = tc["function"]["arguments"]

            print(f"LLM called: {func_name}")
            print(f"Raw JSON:   {arguments_json}")

            if func_name == "final_result":
                # Try to validate the arguments as a MovieReview
                result = output_schema.validate(arguments_json)

                if isinstance(result, str):
                    # Validation failed! result is an error string.
                    # Bounce it back to the LLM so it can fix itself.
                    print(f"Validation FAILED: {result}")

                    # Add the assistant's message + error to history
                    messages.append(response)
                    messages.append(ToolMessage(
                        role="tool",
                        tool_call_id=tc["id"],
                        content=result,
                    ))
                    # Loop continues → LLM tries again!
                else:
                    # Validation succeeded! result is a MovieReview object.
                    print("\nSUCCESS! Got structured output:")
                    print(f"  Title:       {result.title}")
                    print(f"  Rating:      {result.rating}")
                    print(f"  Summary:     {result.summary}")
                    print(f"  Recommended: {result.recommended}")
                    print(f"  Type:        {type(result)}")
                    return
        else:
            # The LLM replied with plain text instead of calling the tool
            print(f"LLM replied with text (unexpected): {response.get('content')}")

    print("\nFailed after max retries.")


if __name__ == "__main__":
    asyncio.run(main())
