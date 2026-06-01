import json

from macro_ds import judge


def test_judge_keeps_high_score(mocker, good_trace):
    mocker.patch.object(
        judge,
        "_call_judge",
        return_value=json.dumps(
            {
                "groundedness": 5,
                "coverage": 4,
                "citation_faithfulness": 5,
                "structure": 4,
                "overall": 4.5,
                "rationale": "well grounded",
            }
        ),
    )
    res = judge.judge_trace(good_trace, model="gpt-x")
    assert res["keep"] is True
    assert res["overall"] >= 4
    assert res["scores"]["groundedness"] == 5


def test_judge_rejects_low_score(mocker, good_trace):
    mocker.patch.object(
        judge,
        "_call_judge",
        return_value=json.dumps(
            {"groundedness": 2, "coverage": 2, "citation_faithfulness": 1, "structure": 3, "overall": 2}
        ),
    )
    res = judge.judge_trace(good_trace, model="gpt-x")
    assert res["keep"] is False


def test_judge_handles_unparseable_output(mocker, good_trace):
    mocker.patch.object(judge, "_call_judge", return_value="not json at all")
    res = judge.judge_trace(good_trace, model="gpt-x")
    assert res["keep"] is False
    assert res["overall"] == 0


def test_judge_prompt_includes_question_and_report(good_trace):
    prompt = judge.build_judge_prompt(good_trace)
    assert good_trace.question in prompt
    assert "Final report" in prompt
