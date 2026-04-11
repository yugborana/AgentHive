from typing import Annotated, Literal
from typing_extensions import TypedDict, NotRequired
import pydantic

class FunctionCall(TypedDict):
    name: str
    arguments: str

class ToolCall(TypedDict):
    id: str
    type: Literal['function']
    function: FunctionCall

class SystemPrompt(TypedDict):
    role: Literal['system']
    content: str

class UserPrompt(TypedDict):
    role: Literal['user']
    content: str

class AssistantMessage(TypedDict):
    role: Literal['assistant']
    content: str | None
    tool_calls: NotRequired[list[ToolCall]]

class ToolMessage(TypedDict):
    role: Literal['tool']
    tool_call_id: str
    content: str

Message = Annotated[
    SystemPrompt | UserPrompt | AssistantMessage | ToolMessage,
    pydantic.Field(discriminator='role')
]

MessageAdapter = pydantic.TypeAdapter(list[Message])
