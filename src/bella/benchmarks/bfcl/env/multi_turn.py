from __future__ import annotations

from typing import Any, Dict, List, Optional

from bella.env.base import EnvironmentSession, ToolCall, ToolResult


class BFCLMultiTurnEnvironmentSession(EnvironmentSession):
    """
    Execute BFCL multi-turn tool calls by directly reusing BFCL backends and
    environment semantics (instances, CLASS_FILE_PATH_MAPPING, etc.).

    Important: This class intentionally does NOT re-implement BFCL backends.
    It delegates execution to BFCL's `execute_multi_turn_func_call`.
    """

    def __init__(
        self,
        *,
        initial_config: Dict[str, Any],
        involved_classes: List[str],
        model_name: str,
        test_entry_id: str,
        long_context: bool = False,
    ) -> None:
        self._initial_config = initial_config or {}
        self._involved_classes = list(involved_classes or [])
        self._model_name = model_name
        self._test_entry_id = test_entry_id
        self._long_context = bool(long_context)

        self._involved_instances: Dict[str, Any] = {}

        # Prime the environment so that instances exist and state is loaded.
        self._prime_instances()

    def _prime_instances(self) -> None:
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
        )

        _, instances = execute_multi_turn_func_call(
            [],
            self._initial_config,
            self._involved_classes,
            self._model_name,
            self._test_entry_id,
            long_context=self._long_context,
            is_evaL_run=False,
        )
        self._involved_instances = instances

    def execute_func_call_strings(
        self, func_calls: List[str]
    ) -> List[str]:
        """
        Execute a list of BFCL-style function call strings (e.g. ["ls(folder='x')"]).
        Returns a list of execution result strings.
        """
        from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
            execute_multi_turn_func_call,
        )

        results, instances = execute_multi_turn_func_call(
            func_calls,
            self._initial_config,
            self._involved_classes,
            self._model_name,
            self._test_entry_id,
            long_context=self._long_context,
            is_evaL_run=False,
        )
        self._involved_instances = instances
        return results

    def execute_one(self, call: ToolCall) -> ToolResult:
        from bella.benchmarks.bfcl.env.tool_executor import render_func_call_string

        func_call_str = render_func_call_string(call.name, call.arguments)
        outputs = self.execute_func_call_strings([func_call_str])
        output = outputs[0] if outputs else ""
        return ToolResult(name=call.name, output=output, tool_call_id=call.tool_call_id)

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a shallow snapshot for debugging (best-effort).
        """
        snap: Dict[str, Any] = {"classes": list(self._involved_instances.keys())}
        for cls, inst in self._involved_instances.items():
            try:
                snap[cls] = {
                    k: v
                    for k, v in vars(inst).items()
                    if not k.startswith("_")
                }
            except Exception:
                continue
        return snap

