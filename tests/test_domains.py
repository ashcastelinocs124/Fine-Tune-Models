import pytest

from macro_ds.domains import GENERAL_SEARCH, MACRO, get_profile


def test_get_profile_resolves_and_fails_fast():
    assert get_profile("macro") is MACRO
    assert get_profile("general_search") is GENERAL_SEARCH
    with pytest.raises(ValueError, match="general_search"):
        get_profile("bogus")


def test_macro_profile_matches_existing_constants():
    """The macro profile must reproduce today's behavior exactly."""
    from macro_ds import judge, prompts, questions

    assert MACRO.system_template == prompts.SYSTEM_TEMPLATE
    assert MACRO.finalize_instruction == prompts.FINALIZE_INSTRUCTION
    assert MACRO.judge_rubric == judge.RUBRIC
    assert MACRO.tool_names is None  # all tools
    assert MACRO.build_system_prompt("2026-07-06", max_steps=12) == prompts.build_system_prompt(
        "2026-07-06", max_steps=12
    )
    assert MACRO.build_question_bank("2026-06-01", 150) == questions.build_question_bank(
        "2026-06-01", 150
    )
    assert MACRO.build_eval_set("2026-06-01", 48) == questions.build_eval_set("2026-06-01", 48)


def test_general_search_prompt_keeps_the_contract_and_drops_fred():
    p = GENERAL_SEARCH.build_system_prompt("2026-07-06", max_steps=10)
    assert "Thought:" in p
    assert "Final report:" in p
    assert "2026-07-06" in p
    assert "10" in p  # max_steps interpolated
    assert "FRED" not in p and "fred" not in p
    assert "macro" not in p.lower()
    assert GENERAL_SEARCH.tool_names == ["web_search", "fetch_url"]
    assert "Final report:" in GENERAL_SEARCH.finalize_instruction
    assert "FRED" not in GENERAL_SEARCH.judge_rubric


def test_general_search_question_bank_is_deterministic_unique_and_big_enough():
    a = GENERAL_SEARCH.build_question_bank("2026-07-06", 100000)
    b = GENERAL_SEARCH.build_question_bank("2026-07-06", 100000)
    assert [q["id"] for q in a] == [q["id"] for q in b]
    assert len({q["question"] for q in a}) == len(a)
    assert len(a) >= 60  # enough headroom for a 25-question Stage 0 draw
    assert all(q["asof_date"] == "2026-07-06" for q in a)


def test_general_search_eval_set_is_disjoint_from_training():
    train = {q["question"] for q in GENERAL_SEARCH.build_question_bank("2026-07-06", 100000)}
    ev = GENERAL_SEARCH.build_eval_set("2026-07-06", 100000)
    ev_qs = {q["question"] for q in ev}
    assert len(ev_qs) == len(ev)
    assert train.isdisjoint(ev_qs)
    assert all(q["id"].startswith("eval-") for q in ev)
