from typing import Annotated, Literal
from typing_extensions import TypedDict
import pydantic
import pydantic_core

class SystemPrompt(TypedDict):
    role : Literal['system']
    content: str

class UserPrompt(TypedDict):
    role: Literal['user']
    content: str

class ToolMessage(TypedDict):
    role: Literal['tool']
    tool_call_id: str
    content: str


class ToolCall(TypedDict):
    id: str
    type: Literal['function']
    function: dict[str, str]

class AssistantMessage(TypedDict):
    role: Literal['assistant']
    content: str | None
    tool_calls: list[ToolCall] | None

# 3. Annotated + Field(discriminator=...)
# We use Annotated to attach Pydantic-specific metadata to a Union type.
# By telling Pydantic to use the 'role' field as a discriminator, it performs 
# "Tagged Union" parsing. When it sees an incoming dict with role='system', 
# it immediately knows to parse it as a SystemPrompt without guessing.
Message = Annotated[
    SystemPrompt | UserPrompt | AssistantMessage | ToolMessage,
    pydantic.Field(discriminator='role')
]

# A TypeAdapter allows us to use Pydantic's validation engine on raw types 
# (like our Message union) without needing them to be inside a BaseModel.
MessageAdapter = pydantic.TypeAdapter(list[Message])

if __name__ == "__main__":
    # Example: A raw list of dictionaries simulating incoming JSON from a database or API
    raw_json_data = [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": "What is the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"}
                }
            ]
        }
    ]

    # Automatically parses and validates the dictionaries into your strictly typed objects!
    validated_messages = MessageAdapter.validate_python(raw_json_data)
    
    print("Successfully validated!")
    for msg in validated_messages:
        print(f"Role: {msg['role']} -> {type(msg)}")