# Agent Design

This document describes the agent architecture used in BELLA's evaluation pipeline.

## Overview

BELLA uses two agent types during evaluation:

- **ReactAgent**: drives the model under test. Loops over LLM calls, detects tool calls via a Model Adapter, executes them against the environment backend, and feeds results back.
- **UserAgent**: simulates a human user for dynamic-mode cases. Generates contextual messages based on persona configuration and signals conversation completion.

Both inherit from a common `BaseAgent` abstract class (inheritance-based design, not layered composition).

## Class Hierarchy

```
BaseAgent (ABC)
├── ReactAgent      # Tool-calling loop for the model under test
└── UserAgent       # LLM-based user simulation for dynamic mode
```

## BaseAgent

Abstract base class providing shared infrastructure:

```python
class BaseAgent(ABC):
    """Base class for all BELLA agents."""

    def __init__(self, provider: str, model_id: str, base_url: str, api_key: str):
        """
        Args:
            provider: "openai" or "anthropic"
            model_id: Model identifier passed to the SDK
            base_url: API endpoint URL
            api_key: Authentication key
        """
        ...
```

Shared responsibilities:
- SDK client management (OpenAI / Anthropic)
- Message format conversion (internal canonical format ↔ SDK-specific format)
- Token usage tracking
- Retry and timeout handling

## ReactAgent

The agent that evaluates the model under test. Receives a `ModelAdapter` via composition.

```python
class ReactAgent(BaseAgent):
    def __init__(self, adapter: ModelAdapter, max_llm_calls: int = 12, **kwargs):
        super().__init__(**kwargs)
        self.adapter = adapter
        self.max_llm_calls = max_llm_calls
```

### Core Loop

```python
def run_turn(self, messages: list[dict], tools: list[dict], backend) -> TurnResult:
    """Execute one conversation turn (may involve multiple LLM calls)."""
    tool_calls_collected = []

    for _ in range(self.max_llm_calls):
        response = self._call_llm(messages, tools)

        if self.adapter.is_tool_call(response):
            parsed = self.adapter.parse_tool_call(response)
            for tc in parsed:
                result = backend.call(tc["name"], tc["arguments"])
                tool_calls_collected.append({
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "result": result
                })
                # Feed tool result back into messages (SDK-specific format)
                self._append_tool_result(messages, tc, result)
        else:
            # No tool call — turn complete
            break

    return TurnResult(
        assistant_message=self._extract_text(response),
        tool_calls=tool_calls_collected,
        messages=messages
    )
```

### Interaction with Model Adapter

The adapter is the ONLY customization point for model-specific behavior:

```
LLM Response → adapter.is_tool_call(response) → bool
             → adapter.parse_tool_call(response) → [{"name": ..., "arguments": ...}]
```

Everything else (sending tools to the model, formatting tool results back) is handled internally by ReactAgent based on the `provider` field.

## Model Adapter

### Protocol

```python
class ModelAdapter(Protocol):
    def is_tool_call(self, response) -> bool:
        """Check if the LLM response contains tool calls."""
        ...

    def parse_tool_call(self, response) -> list[dict]:
        """Extract tool calls from the response.
        Returns: [{"name": "tool_name", "arguments": {"key": "value"}}, ...]
        """
        ...
```

### Built-in Defaults

```python
class OpenAIDefaultAdapter:
    def is_tool_call(self, response) -> bool:
        return response.choices[0].message.tool_calls is not None

    def parse_tool_call(self, response) -> list[dict]:
        return [
            {"name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
            for tc in response.choices[0].message.tool_calls
        ]

class AnthropicDefaultAdapter:
    def is_tool_call(self, response) -> bool:
        return any(block.type == "tool_use" for block in response.content)

    def parse_tool_call(self, response) -> list[dict]:
        return [
            {"name": block.name, "arguments": block.input}
            for block in response.content if block.type == "tool_use"
        ]
```

### Custom Adapter

Users provide a Python file with an `Adapter` class:

```python
# adapters/my_model.py
import json

class Adapter:
    def is_tool_call(self, response) -> bool:
        # Example: model puts tool calls in content as JSON
        content = response.choices[0].message.content or ""
        return "<tool_call>" in content

    def parse_tool_call(self, response) -> list[dict]:
        content = response.choices[0].message.content
        # Parse custom format...
        return [{"name": ..., "arguments": ...}]
```

Specified in config: `model.adapter: adapters/my_model.py`

## UserAgent

Simulates a human user for dynamic-mode cases. Based on Astra's ChatUserAgent design.

```python
class UserAgent(BaseAgent):
    def __init__(self, max_turns: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns
```

### Interface

```python
def start(self, demand: str, user_agent_config: dict) -> UserTurnResult:
    """Generate the first user message based on demand and persona."""
    ...

def respond(self, assistant_message: str) -> UserTurnResult:
    """Generate the next user message in response to the assistant."""
    ...
```

### Return Type

```python
@dataclass
class UserTurnResult:
    message: str      # Generated user message
    is_done: bool     # True when user's goal is achieved or conversation should end
```

### Design Choices

- **Role inversion** (from Astra): system prompt contains persona + demand + rules. The "assistant" role in the LLM call generates user messages, while the "user" role receives the real assistant's responses. This enables better prompt caching.
- **Gradual information reveal**: system prompt instructs the user agent to reveal information progressively (not dump everything in the first message).
- **`[DONE]` signal**: when the user agent determines the goal is achieved (or impossible), it includes `[DONE]` in its output, parsed as `is_done=True`.
- **LLM globally fixed**: the user agent's model is configured in `bella.yaml` under `user_agent`, not per-case. This ensures fair comparison.

## Simulation Orchestration

### Fixed Mode (Track A)

```python
for user_msg in case["user_demands"]:
    messages.append({"role": "user", "content": user_msg})
    result = react_agent.run_turn(messages, tools, backend)
    messages = result.messages
    all_tool_calls.extend(result.tool_calls)
```

### Dynamic Mode (Track B)

```python
user_result = user_agent.start(case["demand"], case["user_agent_config"])
messages.append({"role": "user", "content": user_result.message})

while not user_result.is_done and turn_count < max_turns:
    result = react_agent.run_turn(messages, tools, backend)
    messages = result.messages
    all_tool_calls.extend(result.tool_calls)

    user_result = user_agent.respond(result.assistant_message)
    if not user_result.is_done:
        messages.append({"role": "user", "content": user_result.message})
    turn_count += 1
```

## Dependencies

- `openai` — OpenAI SDK for OpenAI-compatible endpoints (including vLLM)
- `anthropic` — Anthropic SDK for Claude models
- `httpx` — HTTP client for low-level requests if needed

No litellm. The two SDKs cover all supported model providers.
