"""The highest-value test: against the REAL Qwen3 template, ONLY assistant-generated
tokens get loss — never role headers or tool-response framing. Marked `network` because
it downloads the Qwen3-8B tokenizer."""

import pytest

from macro_ds.dataset import trace_to_record
from macro_ds.mask_check import render_and_mask
from macro_ds.schema import Message, ToolCall, Trace

TOKENIZER = "Qwen/Qwen3-8B"


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.mark.network
def test_only_assistant_tokens_get_loss(good_trace):
    tok = _tok()
    rec = trace_to_record(good_trace)
    ids, labels = render_and_mask(rec, tok)

    assert len(ids) == len(labels)
    assert any(l != -100 for l in labels)  # something is trained
    assert any(l == -100 for l in labels)  # something is masked

    unmasked = tok.decode([i for i, l in zip(ids, labels) if l != -100])
    masked = tok.decode([i for i, l in zip(ids, labels) if l == -100])

    # assistant-generated content IS trained
    assert "Thought" in unmasked
    assert "web_search" in unmasked
    assert "Final report" in unmasked

    # system prompt and tool observations are NOT trained
    assert "global-macro" in masked
    assert ("PAGE http" in masked) or ("FRED series" in masked)
    assert "global-macro" not in unmasked

    # CRITICAL: structural/framing tokens must never be trained (the bug the reviewer caught)
    assert "<|im_start|>" not in unmasked  # no role headers trained
    assert "<tool_response>" not in unmasked  # no tool-response framing trained
    assert "user\n" not in unmasked


@pytest.mark.network
def test_mask_holds_on_qwen25_template_with_general_profile():
    """Same guarantee as the Qwen3 test, on the Qwen2.5-7B-Instruct template with a
    general_search (web-only) trace: only assistant tokens get loss."""
    from macro_ds.domains import GENERAL_SEARCH

    msgs = [
        Message(role="system", content=GENERAL_SEARCH.build_system_prompt(asof_date="2026-07-06")),
        Message(role="user", content="What is the state of solar power?"),
        Message(
            role="assistant",
            content="Thought: search for recent solar deployment data.",
            tool_calls=[ToolCall(id="a", name="web_search", arguments={"query": "solar deployment 2026"})],
        ),
        Message(role="tool", content="1. Solar hits record — http://example.com/solar", tool_call_id="a", name="web_search"),
        Message(role="assistant", content="Final report: solar is growing fast [http://example.com/solar]. Findings, uncertainties, and further evidence follow in enough words to clear any length gate."),
    ]
    trace = Trace(question="q", asof_date="2026-07-06", messages=msgs, final_report=msgs[-1].content, steps=2)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    ids, labels = render_and_mask(trace_to_record(trace), tok)

    unmasked = tok.decode([i for i, l in zip(ids, labels) if l != -100])
    masked = tok.decode([i for i, l in zip(ids, labels) if l == -100])

    assert "Thought" in unmasked and "web_search" in unmasked and "Final report" in unmasked
    assert "source-grounded research" in masked  # system prompt not trained
    assert "<|im_start|>" not in unmasked
    assert "<tool_response>" not in unmasked
    assert "user\n" not in unmasked


@pytest.mark.network
def test_multiple_tool_calls_in_one_turn_are_fully_trained():
    """An assistant turn issuing two tool calls at once must train both, and still not
    overshoot into the following tool-response framing."""
    from macro_ds.prompts import build_system_prompt

    msgs = [
        Message(role="system", content=build_system_prompt(asof_date="2026-06-01")),
        Message(role="user", content="What is the 10y yield and the oil price?"),
        Message(
            role="assistant",
            content="Thought: pull both series at once.",
            tool_calls=[
                ToolCall(id="a", name="fred_series", arguments={"series_id": "DGS10"}),
                ToolCall(id="b", name="fred_series", arguments={"series_id": "DCOILWTICO"}),
            ],
        ),
        Message(role="tool", content="FRED series DGS10 (latest 1 obs):\n  2026-05-30  4.50", tool_call_id="a", name="fred_series"),
        Message(role="tool", content="FRED series DCOILWTICO (latest 1 obs):\n  2026-05-30  70.0", tool_call_id="b", name="fred_series"),
        Message(role="assistant", content="Final report: yields 4.5%, oil $70. FRED DGS10, DCOILWTICO. Drivers and risks: many words here to clear the length gate easily."),
    ]
    trace = Trace(question="q", asof_date="2026-06-01", messages=msgs, final_report=msgs[-1].content, steps=2)
    tok = _tok()
    ids, labels = render_and_mask(trace_to_record(trace), tok)
    unmasked = tok.decode([i for i, l in zip(ids, labels) if l != -100])
    assert "DGS10" in unmasked and "DCOILWTICO" in unmasked  # both tool calls trained
    assert "<|im_start|>" not in unmasked  # still no structural overshoot
    assert "<tool_response>" not in unmasked
