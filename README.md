# AgentHive 🐝

**A lightweight, async-first Python framework for building production-ready LLM agents.**

AgentHive gives you a clean, composable toolkit to build agents that can use tools, produce structured output, remember conversations across sessions, stream responses, and delegate work to other agents — all without the overhead of a heavy orchestration framework.

---

## Table of Contents

- [Why AgentHive?](#why-agenthive)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [Messages](#messages)
  - [LLM Providers](#llm-providers)
  - [Tools](#tools)
  - [Structured Output](#structured-output)
  - [Memory & Checkpointing](#memory--checkpointing)
  - [The Agent](#the-agent)
  - [Streaming](#streaming)
  - [Multi-Agent Delegation](#multi-agent-delegation)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Contributing](#contributing)
- [License](#license)

---

## Why AgentHive?

Most agent frameworks are either too opinionated (locking you into their abstractions) or too low-level (leaving you to wire everything yourself). AgentHive hits a deliberate middle ground:

- **Thin abstractions** — the core `Agent` class is a straightforward agentic loop you can read and understand in one sitting.
- **No magic** — tools are plain Python functions; memory is a two-method ABC; the LLM interface is a two-method protocol.
- **Async-first** — everything is `async/await` under the hood, with a `run_sync()` convenience wrapper for scripts and notebooks.
- **Streaming built in** — `run_stream()` lets you yield text chunks token by token while the agentic loop still handles tool calls automatically in the background.
- **Structured output without prompt hacks** — uses the "fake tool" trick to force strict Pydantic-validated JSON from any LLM that supports tool calling.
- **Multi-agent out of the box** — any `Agent` can expose itself as a tool for another agent via `.as_tool()`.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                          Agent                              │
│                                                             │
│   ┌──────────┐   ┌──────────┐   ┌──────────────────────┐   │
│   │   LLM    │   │  Tools   │   │   OutputSchema       │   │
│   │ (Groq,   │   │ (Python  │   │   (Pydantic →        │   │
│   │  etc.)   │   │  funcs)  │   │    fake tool)        │   │
│   └────┬─────┘   └────┬─────┘   └──────────┬───────────┘   │
│        │              │                     │               │
│        └──────────────┴─────────────────────┘               │
│                            │                                │
│                    Agentic Loop                             │
│          (send → receive → tool call → repeat)              │
│                            │                                │
│                    ┌───────┴────────┐                       │
│                    │  MemoryStore   │                       │
│                    │ (SQLite / RAM) │                       │
│                    └───────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

The central loop inside `Agent.run()` follows this flow every turn:

1. Send the full conversation history to the LLM.
2. If the LLM replies with **plain text** → return it (or nudge back if structured output is expected).
3. If the LLM replies with **tool calls** → execute each tool, append the results, loop back to 1.
4. If the tool call is `final_result` (structured output) → validate with Pydantic, return on success or bounce the error back to the LLM on failure.
5. If `max_turns` is exceeded → raise `RuntimeError`.

---

## Project Structure

```
AgentHive/
├── src/
│   ├── __init__.py
│   ├── agent.py          # Agent orchestrator, AgentResult, AgentStream
│   ├── messages.py       # TypedDicts for the OpenAI message format
│   ├── tools.py          # Tool wrapper — turns Python functions into LLM tools
│   ├── result.py         # OutputSchema — structured output via fake tool trick
│   ├── memory.py         # MemoryStore ABC, SQLiteMemoryStore, InMemoryStore
│   └── llm/
│       ├── __init__.py   # LLM base class and ToolSchema dataclass
│       └── groq.py       # Groq provider (LLaMA, Mixtral, etc.)
└── tests/
    └── ...               # Test suite
```

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone the repository
git clone https://github.com/yugborana/AgentHive.git
cd AgentHive

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `groq` | Async Groq SDK (OpenAI-compatible) |
| `pydantic` | Schema building, validation, structured output |
| `aiosqlite` | Async SQLite for the persistent memory store |
| `typing_extensions` | Backports for `TypedDict`, `NotRequired` |

### Environment Variables

Set your Groq API key before running any agent:

```bash
export GROQ_API_KEY="gsk_..."
```

Alternatively, pass `api_key=` directly to `GroqLLM(...)`.

---

## Quick Start

```python
import asyncio
from agenthive import Agent
from agenthive.llm.groq import GroqLLM

# 1. Pick an LLM
llm = GroqLLM("llama-3.3-70b-versatile")

# 2. Define tools as plain Python functions
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        location: The city name, e.g. "Tokyo"
        unit: Temperature unit — "celsius" or "fahrenheit"
    """
    # Replace with a real API call
    return f"Weather in {location}: 22° {unit}"

# 3. Build the agent
agent = Agent(
    llm,
    system_prompt="You are a helpful weather assistant.",
    tools=[get_weather],
)

# 4. Run it
async def main():
    result = await agent.run("What's the weather in Paris?")
    print(result.data)

asyncio.run(main())
```

---

## Core Concepts

### Messages

`messages.py` defines a set of `TypedDict`s that map directly to the OpenAI / Groq chat message format. Using TypedDicts (rather than dataclasses or Pydantic models) means zero serialization overhead — the dicts can be passed straight to the API.

```python
from agenthive.messages import SystemPrompt, UserPrompt, AssistantMessage, ToolMessage

# Every message has a 'role' key used as a discriminator by Pydantic
msg: SystemPrompt = {"role": "system", "content": "You are helpful."}
msg: UserPrompt   = {"role": "user",   "content": "Hello!"}
```

The `Message` union type (with `pydantic.Field(discriminator='role')`) lets Pydantic correctly serialize/deserialize any message variant when saving to memory.

---

### LLM Providers

All providers implement the `LLM` base class with two methods:

```python
class LLM:
    async def chat(
        self, messages: list[Message], tools: list[ToolSchema] | None
    ) -> AssistantMessage: ...

    async def chat_stream(
        self, messages: list[Message], tools: list[ToolSchema] | None
    ) -> StreamResponse: ...
```

#### Groq

```python
from agenthive.llm.groq import GroqLLM

llm = GroqLLM("llama-3.3-70b-versatile")          # reads GROQ_API_KEY from env
llm = GroqLLM("mixtral-8x7b-32768", api_key="...")  # or pass it explicitly
```

Popular Groq models to use:

| Model | Notes |
|---|---|
| `llama-3.3-70b-versatile` | Best general-purpose choice |
| `llama-3.1-8b-instant` | Fastest, lowest latency |
| `mixtral-8x7b-32768` | Large context window |
| `llama3-groq-70b-8192-tool-use-preview` | Optimised for tool use |

**Extending to other providers** (OpenAI, Anthropic, Ollama, etc.) is straightforward — subclass `LLM` and implement `chat()` and `chat_stream()`. Because the `Message` TypedDicts already match the OpenAI wire format, most OpenAI-compatible providers need minimal adaptation.

---

### Tools

AgentHive turns any plain Python function — sync or async — into a tool the LLM can call. No decorators needed.

```python
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web and return a list of results.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
    """
    ...  # your implementation

async def fetch_url(url: str) -> str:
    """Fetch the content of a URL.

    Args:
        url: The full URL to fetch, including https://
    """
    ...  # async implementation is fine too

agent = Agent(llm, tools=[search_web, fetch_url])
```

**How it works internally (`Tool` class):**

1. **Docstring parsing** — The main description and per-parameter descriptions are extracted from Google-style docstrings automatically.
2. **Type hint introspection** — `get_type_hints()` reads the parameter types.
3. **Pydantic model generation** — `pydantic.create_model()` dynamically builds a validation model from the parameters, defaults, and descriptions.
4. **JSON schema** — `.model_json_schema()` generates the schema sent to the LLM.
5. **Argument validation** — when the LLM calls the tool, its JSON arguments are validated against the Pydantic model before the function is invoked. Malformed arguments raise a `ValidationError` that is caught and surfaced cleanly.

```python
# You can also introspect the generated schema directly:
from agenthive.tools import Tool

tool = Tool(search_web)
print(tool.schema)
# ToolSchema(name='search_web', description='Search the web...', parameters={...})
```

---

### Structured Output

When you need the LLM to return data in a specific shape, pass a Pydantic model as `result_type`:

```python
from pydantic import BaseModel
from agenthive import Agent
from agenthive.llm.groq import GroqLLM

class MovieReview(BaseModel):
    title: str
    rating: float          # 0.0–10.0
    summary: str
    recommended: bool

agent = Agent(
    GroqLLM("llama-3.3-70b-versatile"),
    system_prompt="You are a film critic.",
    result_type=MovieReview,
)

result = await agent.run("Review Interstellar.")
review: MovieReview = result.data   # a validated Pydantic model instance

print(review.title)        # "Interstellar"
print(review.rating)       # 9.2
print(review.recommended)  # True
```

**How it works — the "fake tool" trick:**

1. `OutputSchema` converts your Pydantic model into a `ToolSchema` named `final_result`.
2. This schema is appended to the tool list sent to the LLM.
3. The LLM is implicitly forced to call `final_result` (because it's the only way to produce a response when all other tools are available).
4. When the `final_result` call arrives, `OutputSchema.validate()` runs Pydantic validation:
   - **Success** → the loop ends and the validated model instance is returned.
   - **Failure** → a `ToolMessage` containing the formatted Pydantic errors is appended to the conversation, and the LLM gets another chance to fix its output.

This gives you reliable structured output without prompt hacking, regex parsing, or `response_format={"type": "json_object"}`.

---

### Memory & Checkpointing

By default, each `agent.run()` call is stateless. Pass a `MemoryStore` and a `session_id` to persist conversations across calls.

#### SQLite (Persistent)

```python
from agenthive.memory import SQLiteMemoryStore

memory = SQLiteMemoryStore("conversations.db")  # file is created automatically

agent = Agent(llm, memory=memory)

# First call — the agent has no history
result1 = await agent.run("My name is Alice.", session_id="user-001")

# Second call — the agent remembers Alice
result2 = await agent.run("What's my name?", session_id="user-001")
# → "Your name is Alice."
```

SQLite is the default persistent store. It uses `aiosqlite` for fully non-blocking I/O, creates the table and index on first use, and stores each message as a serialized JSON row.

#### In-Memory (Testing)

```python
from agenthive.memory import InMemoryStore

memory = InMemoryStore()  # data lives only for the lifetime of this object
agent = Agent(llm, memory=memory)
```

Useful for unit tests and prototyping. No files, no external dependencies.

#### Custom Memory Backends

Implement the two-method `MemoryStore` ABC to plug in any storage layer — Redis, PostgreSQL, MongoDB, DynamoDB, etc.:

```python
from agenthive.memory import MemoryStore
from agenthive.messages import Message

class RedisMemoryStore(MemoryStore):
    async def get_messages(self, session_id: str) -> list[Message]:
        # fetch from Redis
        ...

    async def add_messages(self, session_id: str, messages: list[Message]) -> None:
        # save to Redis
        ...
```

---

### The Agent

`Agent` is the central orchestrator. All configuration happens in `__init__`; execution happens in `run()`.

```python
from agenthive import Agent

agent = Agent(
    llm,                            # Any LLM provider instance
    system_prompt="...",            # Optional system-level instructions
    tools=[func1, func2],           # Optional list of Python functions
    result_type=MyPydanticModel,    # Optional — enables structured output
    max_turns=10,                   # Max agentic loop iterations (default: 10)
    memory=SQLiteMemoryStore(...),  # Optional memory backend
)
```

**Running an agent:**

```python
# Async (recommended)
result = await agent.run("Your question here", session_id="optional-session-id")

# Sync (convenience wrapper — uses asyncio.run() internally)
result = agent.run_sync("Your question here")

# Access the result
print(result.data)      # str or Pydantic model instance
print(result.messages)  # full list[Message] conversation history
```

**Passing conversation history manually:**

```python
# Continue a conversation without a memory store
history = result.messages
result2 = await agent.run("Follow-up question", message_history=history)
```

---

### Streaming

Use `run_stream()` when you want to display tokens as they arrive — for chat UIs, CLIs, and anything latency-sensitive.

```python
stream = agent.run_stream("Write me a poem about autumn.", session_id="user-001")

# Print tokens as they stream in
async for chunk in stream.stream_text():
    print(chunk, end="", flush=True)

# Get the final AgentResult once streaming is complete
result = await stream.get_data()
```

**How it works:**

`AgentStream` runs the agentic loop in a background `asyncio.Task`. Text chunks are pushed into an `asyncio.Queue` as they arrive from `chat_stream()`. The `stream_text()` async generator drains this queue, yielding each chunk immediately. Tool call fragments are silently accumulated inside `GroqStreamResponse` and assembled into a complete `AssistantMessage` only after the stream ends — so the outer agentic loop can process them normally. A `None` sentinel value is placed in the queue when the loop finishes to signal end-of-stream.

Streaming works correctly alongside tool calls: if the LLM streams text and then calls a tool, the text is streamed to the user and the tool is executed transparently. The next LLM turn (after the tool result) resumes streaming as usual.

---

### Multi-Agent Delegation

Any `Agent` can be wrapped as a callable tool for another agent using `.as_tool()`. This implements the **Agent Delegation** pattern: the parent agent stays in its loop and delegates specific sub-tasks to specialised child agents.

```python
# A specialist agent for code review
code_reviewer = Agent(
    llm,
    system_prompt="You are an expert Python code reviewer. Be concise and precise.",
    result_type=CodeReviewReport,  # optional structured output
)

# A generalist orchestrator agent
orchestrator = Agent(
    llm,
    system_prompt="You are a project manager who coordinates specialist agents.",
    tools=[
        code_reviewer.as_tool(
            name="review_code",
            description="Ask the code reviewer to analyse a Python snippet.",
        ),
        # add more specialist agents as tools here
    ],
)

result = await orchestrator.run(
    "Review this function and summarise the findings:\n\ndef add(a, b):\n    return a - b"
)
```

**How `.as_tool()` works:**

`.as_tool()` generates a regular async Python function named `delegate_agent_tool`. It overwrites `__name__` and `__doc__` so the `Tool` introspection engine builds the correct JSON schema. When the parent LLM calls this tool, the child agent's full `run()` loop executes independently with a fresh conversation — isolating its reasoning from the parent's memory. The child's result (cast to a string if it's a Pydantic model) is returned as a `ToolMessage` to the parent.

---

## API Reference

### `Agent`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `llm` | `LLM` | required | LLM provider instance |
| `system_prompt` | `str` | `""` | System-level instructions prepended to every conversation |
| `tools` | `list[Callable]` | `None` | Python functions to expose as tools |
| `result_type` | `type` | `None` | Pydantic model for structured output |
| `max_turns` | `int` | `10` | Maximum agentic loop iterations before raising `RuntimeError` |
| `memory` | `MemoryStore` | `None` | Memory backend for session persistence |

| Method | Returns | Description |
|---|---|---|
| `run(prompt, message_history, session_id)` | `AgentResult` | Run the agent asynchronously |
| `run_sync(prompt)` | `AgentResult` | Synchronous convenience wrapper |
| `run_stream(prompt, message_history, session_id)` | `AgentStream` | Run with streaming output |
| `as_tool(name, description)` | `Callable` | Expose this agent as a tool for another agent |

### `AgentResult`

| Attribute | Type | Description |
|---|---|---|
| `data` | `str \| BaseModel` | The final answer — plain text or a Pydantic model instance |
| `messages` | `list[Message]` | Full conversation history for this run |

### `AgentStream`

| Method | Returns | Description |
|---|---|---|
| `stream_text()` | `AsyncIterator[str]` | Async generator yielding text chunks |
| `get_data()` | `AgentResult` | Awaitable that resolves once the agent finishes |

### `GroqLLM`

```python
GroqLLM(model: str, *, api_key: str | None = None)
```

Reads `GROQ_API_KEY` from the environment if `api_key` is not provided.

### `SQLiteMemoryStore`

```python
SQLiteMemoryStore(db_path: str = "agent_memory.db")
```

Creates the database file and schema on first use. Thread-safe via `aiosqlite`.

### `InMemoryStore`

```python
InMemoryStore()
```

Dictionary-backed store. All data is lost when the Python process exits.

### `Tool`

```python
Tool(func: Callable)
```

| Attribute / Method | Description |
|---|---|
| `tool.name` | Function name |
| `tool.description` | Parsed from the docstring |
| `tool.schema` | `ToolSchema` ready to send to the LLM |
| `tool.validate_arguments(json_str)` | Parse and validate LLM arguments |
| `await tool.execute(json_str)` | Run the function with validated arguments |

### `OutputSchema`

```python
OutputSchema(response_type: type[T])
```

| Attribute / Method | Description |
|---|---|
| `schema.tool_schema` | The `final_result` `ToolSchema` |
| `schema.validate(json_str)` | Returns `T` on success, error `str` on failure |

---

## Configuration

| Environment Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key. Get one at [console.groq.com](https://console.groq.com) |

For local development, create a `.env` file and load it with `python-dotenv`:

```bash
# .env
GROQ_API_KEY=gsk_...
```

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Running Tests

```bash
# Run the full test suite
pytest tests/

# With verbose output
pytest tests/ -v

# Run a specific test file
pytest tests/test_agent.py
```

---

## Contributing

Contributions are welcome! Here's the recommended workflow:

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feature/my-new-provider
   ```

2. **Make your changes.** Some ideas for contributions:
   - New LLM providers (OpenAI, Anthropic, Ollama, Together AI)
   - New memory backends (Redis, PostgreSQL)
   - Retry logic and exponential backoff
   - Parallel tool execution
   - Token counting and context window management

3. **Write tests** for your changes in the `tests/` directory.

4. **Open a pull request** with a clear description of what you've changed and why.

---

## License

This project is open source. See the [LICENSE](LICENSE) file for details.

---

*Built with 🐝 by [yugborana](https://github.com/yugborana)*
