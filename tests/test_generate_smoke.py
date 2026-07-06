import json

from gen import generate_traces
from macro_ds.schema import Trace


def _write_bank(path, n):
    path.write_text(
        "\n".join(
            json.dumps({"id": f"q{i}", "question": f"question {i}?", "asof_date": "2026-06-01", "theme": "fed"})
            for i in range(n)
        )
    )


def _fake_run_agent(question, asof_date, driver, max_steps=12, profile=None):
    return Trace(
        question=question,
        asof_date=asof_date,
        messages=[],
        final_report="Final report: x [http://u]",
        steps=1,
    )


def test_generate_writes_one_trace_per_question(tmp_path, mocker):
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, 3)
    mocker.patch.object(generate_traces, "run_agent", side_effect=_fake_run_agent)

    n = generate_traces.run_generation(
        str(bank), str(tmp_path / "out"), limit=2, model="gpt-x",
        driver_factory=lambda: object(), concurrency=1,
    )
    assert n == 2
    assert len(list((tmp_path / "out").glob("*.json"))) == 2


def test_generate_is_resumable(tmp_path, mocker):
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, 3)
    out = tmp_path / "out"
    out.mkdir()
    (out / "q0.json").write_text("{}")  # pretend q0 already generated

    calls = []

    def tracking_run_agent(question, asof_date, driver, max_steps=12, profile=None):
        calls.append(question)
        return _fake_run_agent(question, asof_date, driver)

    mocker.patch.object(generate_traces, "run_agent", side_effect=tracking_run_agent)
    n = generate_traces.run_generation(
        str(bank), str(out), limit=3, model="gpt-x", driver_factory=lambda: object(), concurrency=1
    )
    assert n == 2  # q0 skipped, q1 + q2 generated
    assert "question 0?" not in calls


def test_generate_survives_a_failing_trace(tmp_path, mocker):
    bank = tmp_path / "bank.jsonl"
    _write_bank(bank, 2)

    def flaky(question, asof_date, driver, max_steps=12, profile=None):
        if "0" in question:
            raise RuntimeError("boom")
        return _fake_run_agent(question, asof_date, driver)

    mocker.patch.object(generate_traces, "run_agent", side_effect=flaky)
    n = generate_traces.run_generation(
        str(bank), str(tmp_path / "out"), limit=2, model="gpt-x", driver_factory=lambda: object(), concurrency=1
    )
    assert n == 1  # one failed, one succeeded; batch did not crash


def test_run_generation_threads_domain_profile(tmp_path):
    from gen.generate_traces import run_generation
    from macro_ds.schema import Message

    bank = tmp_path / "bank.jsonl"
    bank.write_text(json.dumps({"id": "gs-t-00", "question": "State of solar?",
                                "theme": "t", "asof_date": "2026-07-06"}) + "\n")

    class OneShotDriver:
        def step(self, messages, tool_schemas, allow_tools=True):
            return Message(role="assistant", content="Final report: fine. [http://x]")

    out = tmp_path / "raw"
    written = run_generation(str(bank), str(out), driver_factory=lambda: OneShotDriver(),
                             domain="general_search")
    assert written == 1
    saved = json.loads((out / "gs-t-00.json").read_text())
    system = saved["messages"][0]["content"]
    assert "source-grounded research on any topic" in system
    assert "FRED" not in system
