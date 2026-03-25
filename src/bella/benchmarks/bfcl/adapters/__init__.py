"""BFCL category adapters.

Importing this package triggers ``@register_adapter`` for all categories.
"""
from bella.benchmarks.bfcl.adapters.simple_python import SimplePythonAdapter  # noqa: F401
from bella.benchmarks.bfcl.adapters.multiple import MultipleAdapter  # noqa: F401
from bella.benchmarks.bfcl.adapters.multi_turn_base import MultiTurnBaseAdapter  # noqa: F401
