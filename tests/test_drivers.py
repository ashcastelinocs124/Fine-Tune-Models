from macro_ds.drivers import _messages_to_hf, parse_tool_calls
from macro_ds.schema import Message, ToolCall


def test_parse_single_tool_call_with_thought():
    text = (
        "Thought: I should check the 10y yield.\n"
        '<tool_call>\n{"name": "fred_series", "arguments": {"series_id": "DGS10"}}\n</tool_call>'
    )
    content, calls = parse_tool_calls(text)
    assert content == "Thought: I should check the 10y yield."
    assert len(calls) == 1
    assert calls[0].name == "fred_series"
    assert calls[0].arguments == {"series_id": "DGS10"}


def test_parse_multiple_tool_calls():
    text = (
        "Thought: pull both.\n"
        '<tool_call>\n{"name": "fred_series", "arguments": {"series_id": "DGS10"}}\n</tool_call>\n'
        '<tool_call>\n{"name": "web_search", "arguments": {"query": "fed path"}}\n</tool_call>'
    )
    content, calls = parse_tool_calls(text)
    assert len(calls) == 2
    assert [c.name for c in calls] == ["fred_series", "web_search"]
    assert calls[0].id != calls[1].id


def test_parse_no_tool_calls_is_final_report():
    text = "Final report: yields drift higher. [http://u]"
    content, calls = parse_tool_calls(text)
    assert calls is None
    assert content == text


def test_parse_handles_stringified_arguments():
    text = '<tool_call>\n{"name": "web_search", "arguments": "{\\"query\\": \\"x\\"}"}\n</tool_call>'
    _, calls = parse_tool_calls(text)
    assert calls[0].arguments == {"query": "x"}


def test_messages_to_hf_shapes():
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="q"),
        Message(role="assistant", content="Thought.", tool_calls=[ToolCall(id="1", name="web_search", arguments={"query": "x"})]),
        Message(role="tool", content="res", tool_call_id="1", name="web_search"),
    ]
    hf = _messages_to_hf(msgs)
    assert hf[0] == {"role": "system", "content": "sys"}
    assert hf[2]["tool_calls"][0]["function"]["name"] == "web_search"
    assert hf[2]["tool_calls"][0]["function"]["arguments"] == {"query": "x"}
    assert hf[3]["role"] == "tool" and hf[3]["name"] == "web_search"
