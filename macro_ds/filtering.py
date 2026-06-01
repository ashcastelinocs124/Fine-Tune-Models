"""Mechanical quality gates for rejection sampling.

We distill ONLY from successful trajectories. These deterministic gates run before the
(more expensive) LLM judge and cheaply reject traces that didn't actually do the job:
no final report, no real tool use, no citations, error loops, or repeated identical
calls. A small student learns bad habits fast, so this filter is load-bearing.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from macro_ds.schema import Trace

MIN_REPORT_CHARS = 200
MAX_TOOL_ERRORS = 3
MAX_IDENTICAL_CALLS = 2  # a 3rd identical (name, args) call signals a loop, not a retry
_URL_RE = re.compile(r"https?://", re.IGNORECASE)


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)


def _tool_names_called(trace: Trace) -> list[str]:
    names: list[str] = []
    for m in trace.messages:
        if m.role == "assistant" and m.tool_calls:
            names.extend(tc.name for tc in m.tool_calls)
    return names


def _tool_results(trace: Trace) -> list[str]:
    return [m.content or "" for m in trace.messages if m.role == "tool"]


def _successful_tool(trace: Trace, name: str) -> bool:
    """A tool counts as 'used successfully' if it was called and its result wasn't an ERROR."""
    for m in trace.messages:
        if m.role == "assistant" and m.tool_calls:
            ids = {tc.id for tc in m.tool_calls if tc.name == name}
            if not ids:
                continue
            for r in trace.messages:
                if r.role == "tool" and r.tool_call_id in ids and not (r.content or "").startswith("ERROR:"):
                    return True
    return False


def _series_ids_used(trace: Trace) -> set[str]:
    """FRED series ids the agent actually queried in this trace."""
    ids: set[str] = set()
    for m in trace.messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                if tc.name == "fred_series" and tc.arguments.get("series_id"):
                    ids.add(str(tc.arguments["series_id"]))
    return ids


def _has_citation(report: str, trace: Trace) -> bool:
    """A citation is a URL, or a FRED series id that was ACTUALLY queried in this trace.

    We cross-reference real tool calls instead of scanning prose for acronyms + the word
    "fred" — that heuristic false-passed reports like "GDP rose; the fred database is great".
    """
    if _URL_RE.search(report):
        return True
    return any(sid and sid in report for sid in _series_ids_used(trace))


def _call_counts(trace: Trace) -> Counter:
    counts: Counter = Counter()
    for m in trace.messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                key = (tc.name, json.dumps(tc.arguments, sort_keys=True, default=str))
                counts[key] += 1
    return counts


def _has_loop(trace: Trace) -> bool:
    return any(c > MAX_IDENTICAL_CALLS for c in _call_counts(trace).values())


def _has_malformed_args(trace: Trace) -> bool:
    for m in trace.messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                if "__raw__" in tc.arguments:
                    return True
    return False


def mechanical_gates(trace: Trace) -> GateResult:
    reasons: list[str] = []

    if not trace.completed:
        reasons.append("no final report (incomplete / force-stopped)")
    elif len((trace.final_report or "").strip()) < MIN_REPORT_CHARS:
        reasons.append(f"final report shorter than {MIN_REPORT_CHARS} chars")

    if not _successful_tool(trace, "web_search"):
        reasons.append("no successful web_search call")
    if not _successful_tool(trace, "fetch_url"):
        reasons.append("no successful fetch_url call")

    if trace.completed and not _has_citation(trace.final_report or "", trace):
        reasons.append("final report has no citation (no URL or queried FRED series)")

    if trace.tool_errors > MAX_TOOL_ERRORS:
        reasons.append(f"too many tool errors ({trace.tool_errors} > {MAX_TOOL_ERRORS})")

    if _has_loop(trace):
        reasons.append(f"tool call repeated more than {MAX_IDENTICAL_CALLS}x (possible loop)")

    if _has_malformed_args(trace):
        reasons.append("malformed tool-call arguments (unparseable JSON)")

    return GateResult(passed=not reasons, reasons=reasons)
