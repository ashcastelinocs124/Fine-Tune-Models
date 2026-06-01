from macro_ds.prompts import build_system_prompt


def test_system_prompt_demands_visible_reasoning_and_citations():
    p = build_system_prompt(asof_date="2026-06-01")
    low = p.lower()
    assert "thought:" in low  # visible ReAct reasoning (the hidden-CoT fix)
    assert "2026-06-01" in p  # as-of date embedded
    assert "cite" in low  # grounding/citation requirement
    assert "tool" in low  # tool-use instruction
    assert "report" in low  # final deliverable


def test_system_prompt_mentions_step_budget():
    p = build_system_prompt(asof_date="2026-06-01", max_steps=12)
    assert "12" in p
