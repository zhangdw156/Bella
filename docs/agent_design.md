# Agent Design

This document describes the agent architecture used in BELLA's evaluation pipeline.

## Overview

BELLA uses two agent types during evaluation:

- **ReactAgent**: drives the model under test. Loops over LLM calls, detects tool calls via a Model Adapter, executes them against the environment backend, and feeds results back.
- **UserAgent**: simulates a human user for dynamic-mode cases. Generates contextual messages based on persona configuration and signals conversation completion.

Both inherit from a common `BaseAgent` abstract class (inheritance-based design, not layered composition). Both use Step-based Memory (not raw message lists) for context management.

## Class Hierarchy

```
Data Layer:
  Message(role, content, tool_calls, reasoning_content, token_usage)
  ToolCall(name, arguments, id)
  TokenUsage(input_tokens, output_tokens)
  Timing(start_time, end_time)

Memory Layer:
  Step (base)
  ├── UserStep(content)
  ├── AssistantStep(content, tool_calls, reasoning_content)
  └── ToolResultStep(tool_call_id, name, result)

  Turn:
    user_step: UserStep
    react_steps: list[tuple[AssistantStep, list[ToolResultStep]]]

  Memory:
    system_prompt: str
    turns: list[Turn]

Compaction Layer:
  ContextCompactor (protocol)

Adapter Layer:
  ModelAdapter (protocol)

Agent Layer:
  BaseAgent (ABC)
  ├── ReactAgent
  └── UserAgent

Result Layer:
  TurnResult(assistant_message, tool_calls, reasoning_content, token_usage, timing)
  RunResult(turns, total_tool_calls, token_usage, timing, pass)
```

---

## Data Layer

### Message

```python
@dataclass
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None
    token_usage: TokenUsage | None = None
```

### ToolCall

```python
@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str                               # SDK-generated call ID
```

### TokenUsage

```python
@dataclass
class TokenUsage:
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
```

### Timing

```python
@dataclass
class Timing:
    start_time: float
    end_time: float | None = None

    @property
    def duration(self) -> float | None:
        return None if self.end_time is None else self.end_time - self.start_time
```

---

## Memory Layer

Memory is a list of Steps grouped into Turns. Messages are reconstructed from Steps before each LLM call, giving precise control over what's visible in context.

### Step Types

```python
@dataclass
class UserStep:
    content: str

@dataclass
class AssistantStep:
    content: str                           # Text response
    tool_calls: list[ToolCall]             # Tool calls made (empty if none)
    reasoning_content: str | None = None   # Thinking/reasoning content
    token_usage: TokenUsage | None = None

@dataclass
class ToolResultStep:
    tool_call_id: str
    name: str
    result: dict
```

### Turn

A Turn represents one user message followed by the agent's full response (which may involve multiple LLM calls with tool use).

```python
@dataclass
class Turn:
    user_step: UserStep
    react_steps: list[tuple[AssistantStep, list[ToolResultStep]]]
    timing: Timing | None = None
```

### Memory

```python
class Memory:
    system_prompt: str
    turns: list[Turn]

    def to_messages(self, current_turn_index: int, adapter: ModelAdapter) -> list[Message]:
        """Reconstruct messages from Steps.

        Reasoning visibility rule:
        - Previous turns (index < current_turn_index): reasoning_content STRIPPED
        - Current turn (index == current_turn_index): reasoning_content INCLUDED
          via adapter.format_reasoning()

        Args:
            current_turn_index: Index of the turn currently being processed.
            adapter: Used to format reasoning_content into content when visible.

        Returns:
            Message list ready for LLM call.
        """
        ...

    def estimate_tokens(self) -> int:
        """Estimate total token count using character-based formula."""
        ...
```

### Reasoning Visibility Rules

```
Turn 1 (react loop):
  Step 1: LLM → response (reasoning_content + content + tool_calls)
  Step 2: LLM → context includes Step 1's reasoning_content ✓
  Step 3: LLM → context includes Step 1 & 2's reasoning_content ✓
  → Turn 1 ends

Turn 2 (new user query):
  Step 1: LLM → Turn 1's reasoning_content stripped from context ✗
  Step 2: LLM → context includes Turn 2 Step 1's reasoning_content ✓
```

Within the same turn's react loop: reasoning_content is visible (formatted via `adapter.format_reasoning(content, reasoning_content)`).

Across turns: reasoning_content is stripped. Only `content` from previous turns is included.

---

## Compaction Layer

### ContextCompactor Protocol

```python
class ContextCompactor(Protocol):
    def compact(self, memory: Memory) -> Memory:
        """Compact memory to fit within token budget.

        Invariants:
        - system_prompt is always preserved.
        - The last turn (current) is always preserved.
        - Returns a new Memory instance (does not mutate input).
        """
        ...
```

The compactor is initialized with `max_context_tokens` and optionally a model config (for summarization-based compaction). Token estimation uses a mathematical formula (e.g., character count / 3.5), not a tokenizer.

### Compaction Strategies (pluggable)

- **TruncationCompactor**: drops old turns, truncates tool results (fast, no LLM call)
- **SummarizationCompactor**: calls LLM to summarize old turns (better quality, costs extra)
- **NoopCompactor**: does nothing (for short conversations or testing)

BaseAgent holds a `self.compactor` instance. Before each LLM call, if `memory.estimate_tokens() > max_context_tokens`, compaction is triggered.

---

## Adapter Layer

### ModelAdapter Protocol

```python
class ModelAdapter(Protocol):
    def is_tool_call(self, response) -> bool:
        """Check if the LLM response contains tool calls."""
        ...

    def parse_tool_call(self, response) -> list[dict]:
        """Extract tool calls from the response.
        Returns: [{"name": "...", "arguments": {...}, "id": "..."}]
        """
        ...

    def parse_reasoning(self, response) -> str | None:
        """Extract reasoning_content from the response."""
        ...

    def format_reasoning(self, content: str, reasoning_content: str) -> str:
        """Combine content and reasoning_content into a single string.
        Used when reasoning_content needs to be visible in subsequent LLM calls
        within the same turn.
        """
        ...
```

### Built-in Adapters

```python
class OpenAIChatCompletionsAdapter:
    def is_tool_call(self, response) -> bool:
        return response.choices[0].message.tool_calls is not None

    def parse_tool_call(self, response) -> list[dict]:
        return [
            {"name": tc.function.name, "arguments": json.loads(tc.function.arguments), "id": tc.id}
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
            {"name": block.name, "arguments": block.input, "id": block.id}
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
            {"name": item.name, "arguments": json.loads(item.arguments), "id": item.call_id}
            for item in response.output if item.type == "function_call"
        ]

    def parse_reasoning(self, response) -> str | None:
        reasoning = [item.summary for item in response.output
                     if item.type == "reasoning" and hasattr(item, "summary")]
        return "\n".join(r for r in reasoning if r) if reasoning else None

    def format_reasoning(self, content: str, reasoning_content: str) -> str:
        return f"{reasoning_content}\n\n{content}"
```

### Custom Adapter Example (Qwen3)

```python
# adapters/qwen3.py
class Adapter:
    def is_tool_call(self, response) -> bool:
        return response.choices[0].message.tool_calls is not None

    def parse_tool_call(self, response) -> list[dict]:
        return [
            {"name": tc.function.name, "arguments": json.loads(tc.function.arguments), "id": tc.id}
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

---

## Agent Layer

### BaseAgent

```python
class BaseAgent(ABC):
    def __init__(
        self,
        protocol: str,                    # "anthropic" | "openai_chat_completions" | "openai_responses"
        model_id: str,
        base_url: str,
        api_key: str,
        max_context_tokens: int = 128000,
        compactor: ContextCompactor | None = None,
        temperature: float = 1.0,
    ):
        ...
```

Shared responsibilities:
- SDK client management (selected by `protocol`)
- Message format conversion (Memory → protocol-specific messages)
- Token usage tracking
- Retry and timeout handling
- Auto context compaction (calls `self.compactor.compact(memory)` when token estimate exceeds budget)

### Protocol Support

| Protocol | SDK | Messages Format | Tool Result Format | Reasoning Format |
|----------|-----|----------------|-------------------|-----------------|
| `anthropic` | anthropic | messages + system (separate) | `tool_result` block in user msg | `thinking` content blocks |
| `openai_chat_completions` | openai | messages list | `role: "tool"` with `tool_call_id` | `reasoning_content` field |
| `openai_responses` | openai | items list | `function_call_output` item | reasoning items |

### ReactAgent

```python
class ReactAgent(BaseAgent):
    def __init__(
        self,
        adapter: ModelAdapter,
        max_llm_calls_per_turn: int = 12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.adapter = adapter
        self.max_llm_calls_per_turn = max_llm_calls_per_turn
        self.memory: Memory | None = None
```

#### System Prompt Assembly

The ReactAgent's system prompt is assembled from two parts at runtime:

1. **Common block**: hardcoded in ReactAgent source code — universal behavioral rules.
2. **Category block** (optional): looked up from `category_prompts.json` by the case's `category` field.

Final system prompt = `common_block + "\n\n" + category_block` (if exists).

Both parts are benchmark-controlled — users cannot modify them.

#### Core Loop

```python
def run_turn(self, user_message: str, tools: list[dict], backend) -> TurnResult:
    """Execute one conversation turn (may involve multiple LLM calls)."""
    # Add user step to memory
    turn = Turn(user_step=UserStep(content=user_message), react_steps=[])
    self.memory.turns.append(turn)
    current_turn_idx = len(self.memory.turns) - 1

    for _ in range(self.max_llm_calls_per_turn):
        # Check compaction
        if self.memory.estimate_tokens() > self.max_context_tokens:
            self.memory = self.compactor.compact(self.memory)

        # Reconstruct messages from memory
        messages = self.memory.to_messages(current_turn_idx, self.adapter)

        # Call LLM
        response = self._call_llm(messages, tools)

        # Parse response
        reasoning_content = self.adapter.parse_reasoning(response)
        content = self._extract_text(response)

        if self.adapter.is_tool_call(response):
            parsed_calls = self.adapter.parse_tool_call(response)
            # Execute tools
            tool_results = []
            for tc in parsed_calls:
                result = backend.call(tc["name"], tc["arguments"])
                tool_results.append(ToolResultStep(
                    tool_call_id=tc["id"], name=tc["name"], result=result
                ))
            # Record in memory
            assistant_step = AssistantStep(
                content=content,
                tool_calls=[ToolCall(**tc) for tc in parsed_calls],
                reasoning_content=reasoning_content,
            )
            turn.react_steps.append((assistant_step, tool_results))
        else:
            # No tool call — turn complete
            assistant_step = AssistantStep(content=content, tool_calls=[], reasoning_content=reasoning_content)
            turn.react_steps.append((assistant_step, []))
            break

    return TurnResult(...)
```

### UserAgent

```python
class UserAgent(BaseAgent):
    def __init__(self, max_turns: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.max_turns = max_turns
        self.memory: UserMemory | None = None
```

#### UserAgent Memory

UserAgent uses a simplified memory structure (no tool calls):

```python
@dataclass
class UserAgentStep:
    received_message: str         # Assistant's response (maps to "user" role in LLM call due to role inversion)
    generated_message: str        # Generated user message (maps to "assistant" role in LLM call)
    reasoning_content: str | None = None
    is_done: bool = False

class UserMemory:
    system_prompt: str            # Built from demand + user_agent_config
    steps: list[UserAgentStep]

    def to_messages(self) -> list[Message]:
        """Reconstruct messages with role inversion.
        system: persona + demand + rules
        assistant: generated user messages
        user: received assistant responses
        """
        ...
```

#### Interface

```python
def start(self, demand: str, user_agent_config: dict) -> UserTurnResult:
    """Initialize memory with system prompt and generate first user message.

    System prompt is built from:
    - demand: what the user wants to achieve
    - user_agent_config.role: who the user is
    - user_agent_config.personality: how the user behaves
    - user_agent_config.knowledge_boundary: what the user knows/doesn't know
    """
    ...

def respond(self, assistant_message: str) -> UserTurnResult:
    """Generate the next user message in response to the assistant."""
    ...
```

#### Return Type

```python
@dataclass
class UserTurnResult:
    message: str      # Generated user message
    is_done: bool     # True when goal achieved, impossible, or conversation should end
```

#### Design Choices

- **Role inversion**: system prompt contains persona + demand + rules. The "assistant" role generates user messages, while the "user" role receives assistant responses. Enables prompt caching.
- **Gradual information reveal**: system prompt instructs progressive disclosure (first message = primary reason only, reveal details when asked).
- **`[DONE]` signal**: output contains `[DONE]` → `is_done=True`. Triggered when goal is achieved, goal is impossible, or assistant transfers to human.
- **Knowledge boundary enforcement**: system prompt explicitly states what to know and not know. If asked about unknown info, respond with "I don't know" or a vague answer.
- **Context compaction**: inherits BaseAgent's auto-compaction for long conversations.
- **LLM globally fixed**: model configured in `bella.yaml` under `user_agent`, not per-case.

---

## Simulation Orchestration

### Fixed Mode (Track A)

```python
def run_fixed(case, react_agent, tools, backend) -> RunResult:
    react_agent.init_memory(system_prompt)

    for user_msg in case["user_demands"]:
        result = react_agent.run_turn(user_msg, tools, backend)

    return RunResult(...)
```

### Dynamic Mode (Track B)

```python
def run_dynamic(case, react_agent, user_agent, tools, backend) -> RunResult:
    react_agent.init_memory(system_prompt)

    user_result = user_agent.start(case["demand"], case["user_agent_config"])

    while not user_result.is_done and turn_count < max_turns:
        result = react_agent.run_turn(user_result.message, tools, backend)
        user_result = user_agent.respond(result.assistant_message)
        turn_count += 1

    return RunResult(...)
```

---

## Result Layer

### TurnResult

```python
@dataclass
class TurnResult:
    assistant_message: str
    tool_calls: list[ToolCall]
    reasoning_content: str | None
    token_usage: TokenUsage | None
    timing: Timing | None
```

### RunResult

```python
@dataclass
class RunResult:
    turns: list[Turn]
    total_tool_calls: list[ToolCall]     # Flattened from all turns
    token_usage: TokenUsage              # Aggregated
    timing: Timing
    ended_normally: bool                 # UserAgent signaled [DONE] or all user_demands exhausted
```

---

## Dependencies

- `openai` — OpenAI SDK for chat completions and responses API (including vLLM)
- `anthropic` — Anthropic SDK for Claude models
- `httpx` — HTTP client for low-level requests if needed

No litellm. The two SDKs cover all supported model providers.
