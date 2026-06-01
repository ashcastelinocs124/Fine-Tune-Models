"""Mechanical eval metrics computed from a Trace.

These are the cheap, deterministic metrics in the three-way eval (base vs student vs
teacher): they measure whether the agent operated the tools correctly and grounded its
numbers, independent of the LLM-judge quality scores.
"""

from __future__ import annotations

import re
from typing import Any

from macro_ds.schema import Trace
from macro_ds.tools import TOOLS

_VALID_TOOL_NAMES = set(TOOLS)
# Numbers like 4.5, 70, 99.9, 2026, 1,200 (commas stripped before matching).
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _all_tool_calls(trace: Trace):
    for m in trace.messages:
        if m.role == "assistant" and m.tool_calls:
            yield from m.tool_calls


def mechanical_metrics(trace: Trace) -> dict[str, Any]:
    calls = list(_all_tool_calls(trace))
    n = len(calls)
    valid = sum(
        1 for tc in calls if tc.name in _VALID_TOOL_NAMES and "__raw__" not in tc.arguments
    )
    return {
        "completion": trace.completed,
        "steps": trace.steps,
        "n_tool_calls": n,
        "valid_tool_call_rate": (valid / n) if n else 0.0,
        "tool_error_rate": (trace.tool_errors / n) if n else 0.0,
    }


def hallucinated_number_rate(trace: Trace) -> float:
    """Fraction of numeric claims in the final report not present in any retrieved tool output.

    Heuristic (substring match against concatenated tool results). Commas are stripped so
    "1,200" matches "1200". A high rate flags invented figures.
    """
    report = (trace.final_report or "").replace(",", "")
    numbers = _NUMBER_RE.findall(report)
    if not numbers:
        return 0.0
    retrieved = "".join(m.content or "" for m in trace.messages if m.role == "tool").replace(",", "")
    ungrounded = sum(1 for num in numbers if num not in retrieved)
    return ungrounded / len(numbers)
