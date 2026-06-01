import json

import pytest

from macro_ds.dataset import trace_to_record


def test_trace_converts_to_messages_record(good_trace):
    rec = trace_to_record(good_trace)
    roles = [m["role"] for m in rec["messages"]]
    assert roles[0] == "user"  # system goes in rec["system"], not messages
    assert "assistant" in roles and "tool" in roles
    assert rec["system"].startswith("You are a global-macro")
    # tools attached as a JSON string of the 4 tool schemas
    tools = json.loads(rec["tools"])
    assert {t["function"]["name"] for t in tools} == {"web_search", "fetch_url", "fred_series", "fred_search"}


def test_assistant_tool_turn_keeps_thought_and_call(good_trace):
    rec = trace_to_record(good_trace)
    ac = next(m for m in rec["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
    assert ac["content"].startswith("Thought")  # reasoning preserved alongside the call
    assert ac["tool_calls"][0]["function"]["name"] in {"web_search", "fetch_url", "fred_series", "fred_search"}


def test_tool_messages_have_tool_call_id(good_trace):
    rec = trace_to_record(good_trace)
    tool_msgs = [m for m in rec["messages"] if m["role"] == "tool"]
    assert tool_msgs and all(m.get("tool_call_id") for m in tool_msgs)


def test_incomplete_trace_is_rejected(no_report_trace):
    with pytest.raises(ValueError):
        trace_to_record(no_report_trace)


def test_build_split_drops_incomplete_and_is_deterministic(good_trace, no_report_trace):
    from macro_ds.dataset import build_split

    traces = [good_trace] * 9 + [no_report_trace]  # 9 valid + 1 incomplete
    train, val = build_split(traces, val_frac=0.1, seed=0)
    assert len(train) + len(val) == 9  # incomplete dropped
    assert len(val) == 1  # int(9*0.1)=0 -> floored up to 1
    assert len(train) == 8

    train2, val2 = build_split(traces, val_frac=0.1, seed=0)
    assert [r["messages"] for r in train] == [r["messages"] for r in train2]  # deterministic
