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

    def __init__(self, protocol: str, model_id: str, base_url: str, api_key: str,
                 max_context_tokens: int = 128000):
        """
        Args:
            protocol: "anthropic", "openai_chat_completions", or "openai_responses"
            model_id: Model identifier passed to the SDK
            base_url: API endpoint URL
            api_key: Authentication key
            max_context_tokens: Maximum context window size for auto-compaction
        """
        ...
```

Shared responsibilities:
- SDK client management (OpenAI / Anthropic, selected by `protocol`)
- Message format conversion (internal canonical format <-> protocol-specific format)
- Token usage tracking
- Retry and timeout handling
- **Auto context compaction** (see below)

### Protocol Support

| Protocol | SDK | Messages Format | Tool Result Format | Reasoning Format |
|----------|-----|----------------|-------------------|-----------------|
| `anthropic` | anthropic | messages + system (separate) | `tool_result` block in user msg | `thinking` content blocks |
| `openai_chat_completions` | openai | messages list | `role: "tool"` with `tool_call_id` | `reasoning_content` field |
| `openai_responses` | openai | items list | `function_call_output` item | reasoning items |

### Auto Context Compaction

When the message context approaches `max_context_tokens`, BaseAgent automatically compresses older messages. The compaction strategy (inspired by Qwen-Agent's multi-phase truncation):

1. **Phase 1**: Truncate tool result content in older turns (keep tool name and status, drop verbose output)
2. **Phase 2**: Drop entire middle conversation turns (preserve first and last turns)
3. **Phase 3**: Truncate remaining long messages, keeping both beginning and end

Both ReactAgent and UserAgent inherit this capability.

## ReactAgent

The agent that evaluates the model under test. Receives a `ModelAdapter` via composition.

### System Prompt Assembly

The ReactAgent's system prompt is assembled from two parts at runtime:

1. **Common block**: hardcoded in ReactAgent source code — universal behavioral rules.
2. **Category block** (optional): looked up from `category_prompts.json` by the case's `category` field. Contains domain-specific business rules and policies.

Final system prompt = `common_block + "\n\n" + category_block` (if category block exists).

Both parts are benchmark-controlled — users cannot modify them. Temperature defaults to 1.0 but is user-configurable.

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

        # Extract reasoning (if any)
        reasoning = self.adapter.parse_reasoning(response)

        if self.adapter.is_tool_call(response):
            parsed = self.adapter.parse_tool_call(response)
            for tc in parsed:
                result = backend.call(tc["name"], tc["arguments"])
                tool_calls_collected.append({
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "result": result
                })
            # Append assistant message WITH reasoning (visible within this turn)
            self._append_assistant_message(messages, response, reasoning, include_reasoning=True)
            # Append tool results
            for tc in tool_calls_collected[-len(parsed):]:
                self._append_tool_result(messages, tc)
        else:
            # No tool call — turn complete
            break

    return TurnResult(
        assistant_message=self._extract_text(response),
        tool_calls=tool_calls_collected,
        reasoning=reasoning,
        messages=messages
    )
```

### Reasoning Visibility Rules

Reasoning content follows a **same-turn visible, cross-turn invisible** policy:

```
Turn 1 (react loop):
  Step 1: LLM → response (reasoning + content + tool_calls)
  Step 2: LLM → context includes Step 1's reasoning ✓
  Step 3: LLM → context includes Step 1 & 2's reasoning ✓
  → Turn 1 ends

Turn 2 (new user query):
  Step 1: LLM → Turn 1's reasoning stripped from context ✗
```

**Within the same turn's react loop**: reasoning content is included in the assistant message via `adapter.format_reasoning()`, so subsequent LLM calls in the same loop can see it.

**Across turns**: when building context for a new turn (after a new user message), previous turns' reasoning content is stripped from assistant messages. Only `content` is preserved.

### Interaction with Model Adapter

The adapter handles all model-specific output parsing:

```
LLM Response → adapter.is_tool_call(response) → bool
             → adapter.parse_tool_call(response) → [{"name": ..., "arguments": ...}]
             → adapter.parse_reasoning(response) → str | None
             → adapter.format_reasoning(content, reasoning_content) → str
```

Everything else (sending tools to the model, formatting tool results back, context management) is handled internally by ReactAgent based on the `protocol` field.

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

    def parse_reasoning(self, response) -> str | None:
        """Extract reasoning/thinking content from the response.
        Returns the reasoning text, or None if no reasoning is present.
        """
        ...

    def format_reasoning(self, content: str, reasoning_content: str) -> str:
        """Combine content and reasoning_content into a single string for context.
        Used when reasoning needs to be visible in subsequent LLM calls within the same turn.
        """
        ...
```

### Built-in Defaults

```python
class OpenAIChatCompletionsAdapter:
    def is_tool_call(self, response) -> bool:
        return response.choices[0].message.tool_calls is not None

    def parse_tool_call(self, response) -> list[dict]:
        return [
            {"name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
            for tc in response.choices[0].message.tool_calls
        ]

    def parse_reasoning(self, response) -> str | None:
        return getattr(response.choices[0].message, "reasoning_content", None)

    def format_reasoning(self, content: str, reasoning_content: str) -> str:
        return f"{reasoning_content}\n\n{content}"


class AnthropicAdapter:
    def is_tool_call(self, response) -> bool:
        return any(block.type == "tool_use" for block in response.content)

    def parse_tool_call(self, response) -> list[dict]:
        return [
            {"name": block.name, "arguments": block.input}
            for block in response.content if block.type == "tool_use"
        ]

    def parse_reasoning(self, response) -> str | None:
        thinking = [block.thinking for block in response.content if block.type == "thinking"]
        return "\n".join(thinking) if thinking else None

    def format_reasoning(self, content: str, reasoning_content: str) -> str:
        return f"<thinking>{reasoning_content}</thinking>\n{content}"


class OpenAIResponsesAdapter:
    def is_tool_call(self, response) -> bool:
        return any(item.type == "function_call" for item in response.output)

    def parse_tool_call(self, response) -> list[dict]:
        return [
            {"name": item.name, "arguments": json.loads(item.arguments)}
            for item in response.output if item.type == "function_call"
        ]

    def parse_reasoning(self, response) -> str | None:
        reasoning = [item.summary for item in response.output
                     if item.type == "reasoning" and hasattr(item, "summary")]
        return "\n".join(r for r in reasoning if r) if reasoning else None

    def format_reasoning(self, content: str, reasoning_content: str) -> str:
        return f"{reasoning_content}\n\n{content}"
```

### Custom Adapter

Users provide a Python file with an `Adapter` class. Only override the methods that differ from the default:

```python
# adapters/qwen3.py

class Adapter:
    def is_tool_call(self, response) -> bool:
        return response.choices[0].message.tool_calls is not None

    def parse_tool_call(self, response) -> list[dict]:
        return [
            {"name": tc.function.name, "arguments": json.loads(tc.function.arguments)}
            for tc in response.choices[0].message.tool_calls
        ]

    def parse_reasoning(self, response) -> str | None:
        content = response.choices[0].message.content or ""
        if "<think>" in content and "</think>" in content:
            start = content.index("<think>") + len("<think>")
            end = content.index("</think>")
            return content[start:end].strip()
        return None

    def format_reasoning(self, content: str, reasoning_content: str) -> str:
        return f"<think>{reasoning_content}</think>\n{content}"
```

Specified in config: `model.adapter: adapters/qwen3.py`

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
- **Context compaction**: inherits BaseAgent's auto-compaction for long conversations.

## Simulation Orchestration

### Fixed Mode (Track A)

```python
for user_msg in case["user_demands"]:
    messages.append({"role": "user", "content": user_msg})
    result = react_agent.run_turn(messages, tools, backend)
    # Strip reasoning from this turn's messages before next turn
    strip_reasoning_from_last_turn(messages)
    all_tool_calls.extend(result.tool_calls)
```

### Dynamic Mode (Track B)

```python
user_result = user_agent.start(case["demand"], case["user_agent_config"])
messages.append({"role": "user", "content": user_result.message})

while not user_result.is_done and turn_count < max_turns:
    result = react_agent.run_turn(messages, tools, backend)
    # Strip reasoning from this turn's messages before next turn
    strip_reasoning_from_last_turn(messages)
    all_tool_calls.extend(result.tool_calls)

    user_result = user_agent.respond(result.assistant_message)
    if not user_result.is_done:
        messages.append({"role": "user", "content": user_result.message})
    turn_count += 1
```

## Dependencies

- `openai` — OpenAI SDK for chat completions and responses API (including vLLM)
- `anthropic` — Anthropic SDK for Claude models
- `httpx` — HTTP client for low-level requests if needed

No litellm. The two SDKs cover all supported model providers.
