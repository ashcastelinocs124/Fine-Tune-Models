"""Convert normalized traces into training records.

Output is the OpenAI-style `messages` + `tools` + `system` shape that LLaMA-Factory
ingests for multi-turn tool-calling SFT. The assistant turn that calls a tool keeps
BOTH its `content` (the visible "Thought:" reasoning) and its `tool_calls`, so the
student is trained to reason *and* act — that reasoning is the distillation target.
"""

from __future__ import annotations

import json
import random
from typing import Any

from macro_ds.schema import Trace
from macro_ds.tools import openai_tool_schemas


def trace_to_record(trace: Trace) -> dict[str, Any]:
    """Convert one completed trace to a LLaMA-Factory training record.

    Raises ValueError on incomplete/force-stopped traces (defensive — only successful
    trajectories should reach training).
    """
    if not trace.completed:
        raise ValueError("cannot convert an incomplete trace (no final report)")

    system = ""
    messages: list[dict[str, Any]] = []
    for m in trace.messages:
        if m.role == "system":
            system = m.content or ""
            continue
        messages.append(m.to_openai())

    return {
        "messages": messages,
        "system": system,
        "tools": json.dumps(openai_tool_schemas()),
    }


def build_split(
    traces: list[Trace], val_frac: float = 0.1, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert traces to records (dropping incomplete ones) and split deterministically."""
    records: list[dict[str, Any]] = []
    for t in traces:
        try:
            records.append(trace_to_record(t))
        except ValueError:
            continue  # incomplete trace — skip defensively

    random.Random(seed).shuffle(records)
    if len(records) <= 1:
        return records, []

    n_val = max(1, int(len(records) * val_frac))
    n_val = min(n_val, len(records) - 1)  # always keep at least 1 train record
    return records[n_val:], records[:n_val]
