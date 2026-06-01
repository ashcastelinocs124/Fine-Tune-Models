from macro_ds.metrics import hallucinated_number_rate, mechanical_metrics
from macro_ds.schema import ToolCall


def test_mechanical_metrics_basic(good_trace):
    m = mechanical_metrics(good_trace)
    assert m["completion"] is True
    assert m["n_tool_calls"] == 3
    assert m["tool_error_rate"] == 0.0
    assert m["valid_tool_call_rate"] == 1.0  # web_search, fetch_url, fred_series all known + well-formed
    assert m["steps"] == good_trace.steps


def test_valid_tool_call_rate_penalizes_unknown_and_malformed(good_trace):
    good_trace.messages[2].tool_calls.append(ToolCall(id="x", name="bogus_tool", arguments={}))
    good_trace.messages[4].tool_calls.append(ToolCall(id="y", name="web_search", arguments={"__raw__": "{bad"}))
    m = mechanical_metrics(good_trace)
    assert m["n_tool_calls"] == 5
    assert m["valid_tool_call_rate"] == 0.6  # 3 of 5 valid


def test_hallucinated_number_rate_zero_when_grounded(good_trace):
    # report cites 4.50 which appears in the FRED tool result; date digits also appear there
    assert hallucinated_number_rate(good_trace) == 0.0


def test_hallucinated_number_rate_detects_ungrounded(good_trace):
    rep = good_trace.final_report + " Also, unemployment is 99.9%."
    good_trace.final_report = rep
    good_trace.messages[-1].content = rep
    assert hallucinated_number_rate(good_trace) > 0.0  # 99.9 not in any tool output


def test_hallucinated_number_rate_no_numbers_is_zero():
    from macro_ds.schema import Message, Trace

    rep = "Final report: yields will rise on sticky inflation. [http://u]"
    t = Trace(
        question="q",
        asof_date="2026-06-01",
        messages=[Message(role="assistant", content=rep)],
        final_report=rep,
        steps=1,
    )
    assert hallucinated_number_rate(t) == 0.0
