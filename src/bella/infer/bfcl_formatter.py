from __future__ import annotations

from typing import Any, Dict, List, Tuple


def build_simple_python_request(entry: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build OpenAI chat messages and tools for a BFCL simple_python entry.

    This is a deliberately simple formatter:
    - One system message describing the task
    - One user message using the BFCL `question` content as-is (JSON string)
    - Tools schema derived directly from BFCL `function` definitions
    """
    test_id = entry.get("id", "")
    question = entry.get("question", [])
    functions = entry.get("function", [])

    # System prompt: very lightweight, we don't try to match BFCL's internal prompts.
    system_content = (
        "You are a function calling model. "
        "Given a user question and a set of available functions, "
        "you must choose the most appropriate function and provide arguments "
        "as a JSON object that satisfies the given JSON schema."
    )

    # 从 BFCL question 里提取第一条用户自然语言问题（simple_python 是单轮 non-live）。
    if isinstance(question, list) and question and isinstance(question[0], list):
        first_turn = question[0]
        if first_turn and isinstance(first_turn[0], dict):
            user_text = first_turn[0].get("content", "")
        else:
            user_text = str(question)
    else:
        user_text = str(question)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"BFCL test id: {test_id}\n"
                f"User question:\n{user_text}\n\n"
                "Choose ONE function from the provided tools and provide only JSON arguments."
            ),
        },
    ]

    tools: List[Dict[str, Any]] = []
    for func in functions:
        # BFCL simple_python functions should already carry JSON schema info.
        original_name = func.get("name")
        description = func.get("description", "")
        parameters = func.get("parameters", {})

        if not original_name or not parameters:
            # Skip malformed entries; evaluator会认为该条无效/错误
            continue

        # BFCL evaluator 对 simple_python 的 FC 模型通常期望使用下划线风格的函数名，
        # 例如 math_factorial / math_hypot，而不是 math.factorial / math.hypot。
        # 这里对 BFCL 原始函数名做一个最小映射：用下划线替换点号。
        tool_name = original_name.replace(".", "_")

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    return messages, tools

