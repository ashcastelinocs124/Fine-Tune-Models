"""Shared synthetic-trace fixtures for filtering / judge / dataset tests."""

import pytest

from macro_ds.prompts import build_system_prompt
from macro_ds.schema import Message, ToolCall, Trace

SYSTEM = build_system_prompt(asof_date="2026-06-01")


def _base_messages(report: str | None):
    msgs = [
        Message(role="system", content=SYSTEM),
        Message(role="user", content="What is the path of the US 10y yield?"),
        Message(
            role="assistant",
            content="Thought: search for the latest 10y yield commentary.",
            tool_calls=[ToolCall(id="1", name="web_search", arguments={"query": "US 10y yield outlook"})],
        ),
        Message(role="tool", content="[1] Yields rise\n    http://news/yields\n    snippet", tool_call_id="1", name="web_search"),
        Message(
            role="assistant",
            content="Thought: read the top source.",
            tool_calls=[ToolCall(id="2", name="fetch_url", arguments={"url": "http://news/yields"})],
        ),
        Message(role="tool", content="PAGE http://news/yields\nThe 10y yield rose to 4.5%.", tool_call_id="2", name="fetch_url"),
        Message(
            role="assistant",
            content="Thought: confirm with FRED data.",
            tool_calls=[ToolCall(id="3", name="fred_series", arguments={"series_id": "DGS10"})],
        ),
        Message(role="tool", content="FRED series DGS10 (latest 2 obs):\n  2026-05-30  4.50", tool_call_id="3", name="fred_series"),
    ]
    if report is not None:
        msgs.append(Message(role="assistant", content=report))
    return msgs


@pytest.fixture
def good_trace() -> Trace:
    report = (
        "Final report: The US 10y yield is likely to drift modestly higher. "
        "FRED DGS10 shows 4.50% as of 2026-05-30 [http://news/yields]. "
        "Key drivers: sticky inflation and heavy issuance. Risks: a growth scare."
    )
    return Trace(
        question="What is the path of the US 10y yield?",
        asof_date="2026-06-01",
        messages=_base_messages(report),
        final_report=report,
        steps=3,
        tool_errors=0,
        meta={"model": "gpt-x"},
    )


@pytest.fixture
def no_report_trace() -> Trace:
    return Trace(
        question="What is the path of the US 10y yield?",
        asof_date="2026-06-01",
        messages=_base_messages(None),
        final_report=None,
        steps=12,
        tool_errors=0,
    )


@pytest.fixture
def uncited_trace() -> Trace:
    report = "Final report: Yields go up. Inflation is sticky. That is my view."
    return Trace(
        question="What is the path of the US 10y yield?",
        asof_date="2026-06-01",
        messages=_base_messages(report),
        final_report=report,
        steps=3,
        tool_errors=0,
    )
