import json

from macro_ds.schema import Message, ToolCall, Trace


def test_assistant_message_with_tool_call_roundtrips_to_openai():
    m = Message(
        role="assistant",
        content="Thought: check yields.",
        tool_calls=[ToolCall(id="c1", name="web_search", arguments={"query": "10y yield"})],
    )
    oai = m.to_openai()
    assert oai["role"] == "assistant"
    assert oai["content"] == "Thought: check yields."
    assert oai["tool_calls"][0]["id"] == "c1"
    assert oai["tool_calls"][0]["type"] == "function"
    assert oai["tool_calls"][0]["function"]["name"] == "web_search"
    assert json.loads(oai["tool_calls"][0]["function"]["arguments"]) == {"query": "10y yield"}


def test_assistant_message_from_openai_roundtrips():
    raw = {
        "role": "assistant",
        "content": "Thought: look it up.",
        "tool_calls": [
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "fred_series", "arguments": json.dumps({"series_id": "DGS10"})},
            }
        ],
    }
    m = Message.from_openai(raw)
    assert m.role == "assistant"
    assert m.content == "Thought: look it up."
    assert m.tool_calls[0].name == "fred_series"
    assert m.tool_calls[0].arguments == {"series_id": "DGS10"}
    # round trip back out is equivalent
    assert m.to_openai()["tool_calls"][0]["function"]["name"] == "fred_series"


def test_tool_message_serialization_preserves_name():
    m = Message(role="tool", content="RESULT: 4.5", tool_call_id="c1", name="fred_series")
    oai = m.to_openai()
    assert oai == {"role": "tool", "tool_call_id": "c1", "content": "RESULT: 4.5", "name": "fred_series"}


def test_tool_message_without_name_omits_key():
    m = Message(role="tool", content="x", tool_call_id="c1")
    assert "name" not in m.to_openai()


def test_plain_assistant_message_has_no_tool_calls_key():
    m = Message(role="assistant", content="Final report: yields up.")
    oai = m.to_openai()
    assert oai == {"role": "assistant", "content": "Final report: yields up."}


def test_trace_json_roundtrip():
    t = Trace(
        question="Where are yields headed?",
        asof_date="2026-06-01",
        messages=[
            Message(role="system", content="You are..."),
            Message(role="user", content="Where are yields headed?"),
            Message(
                role="assistant",
                content="Thought: search.",
                tool_calls=[ToolCall(id="1", name="web_search", arguments={"query": "x"})],
            ),
            Message(role="tool", content="RESULT: ...", tool_call_id="1", name="web_search"),
            Message(role="assistant", content="Final report: up. [http://u]"),
        ],
        final_report="Final report: up. [http://u]",
        steps=2,
        tool_errors=0,
        usage={"prompt_tokens": 10, "completion_tokens": 5},
        meta={"model": "gpt-x"},
    )
    s = t.to_json()
    assert isinstance(s, str)
    t2 = Trace.from_json(s)
    assert t2.question == t.question
    assert len(t2.messages) == 5
    assert t2.messages[2].tool_calls[0].name == "web_search"
    assert t2.final_report == t.final_report
    assert t2.completed is True


def test_trace_completed_flag_false_when_no_report():
    t = Trace(question="q", asof_date="2026-06-01", messages=[], final_report=None, steps=12)
    assert t.completed is False
