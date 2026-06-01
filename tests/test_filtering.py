from macro_ds.filtering import mechanical_gates


def test_gates_pass_good_trace(good_trace):
    res = mechanical_gates(good_trace)
    assert res.passed, res.reasons


def test_gates_reject_incomplete(no_report_trace):
    res = mechanical_gates(no_report_trace)
    assert not res.passed
    assert any("report" in r.lower() or "complete" in r.lower() for r in res.reasons)


def test_gates_reject_uncited(uncited_trace):
    res = mechanical_gates(uncited_trace)
    assert not res.passed
    assert any("cit" in r.lower() for r in res.reasons)


def test_gates_reject_when_no_fetch(good_trace):
    # drop the fetch_url tool messages -> should fail "used fetch" gate
    good_trace.messages = [
        m for m in good_trace.messages if not (m.name == "fetch_url" or (m.tool_calls and m.tool_calls[0].name == "fetch_url"))
    ]
    res = mechanical_gates(good_trace)
    assert not res.passed
    assert any("fetch" in r.lower() for r in res.reasons)


def test_gates_reject_too_many_tool_errors(good_trace):
    good_trace.tool_errors = 99
    res = mechanical_gates(good_trace)
    assert not res.passed
    assert any("error" in r.lower() for r in res.reasons)


def test_gates_reject_acronym_without_real_citation(good_trace):
    # Looks "cited" to a naive heuristic (has acronyms + the word "fred") but cites no URL
    # and no FRED series the agent actually queried -> must be rejected.
    bad = (
        "Final report: GDP is rising and CPI is sticky. The fred database is great. "
        + "Lots more macro analysis here to comfortably clear the minimum length gate. " * 4
    )
    good_trace.final_report = bad
    good_trace.messages[-1].content = bad
    res = mechanical_gates(good_trace)
    assert not res.passed
    assert any("cit" in r.lower() for r in res.reasons)


def test_gates_accept_real_series_citation_without_url(good_trace):
    # No URL, but cites DGS10 which the trace actually queried -> counts as cited.
    rep = "Final report: The 10y yield per DGS10 is 4.5%. " + "Drivers and risks discussed here. " * 6
    good_trace.final_report = rep
    good_trace.messages[-1].content = rep
    assert mechanical_gates(good_trace).passed


def test_gates_reject_malformed_tool_args(good_trace):
    good_trace.messages[2].tool_calls[0].arguments = {"__raw__": "{bad json"}
    res = mechanical_gates(good_trace)
    assert not res.passed
    assert any("malformed" in r.lower() for r in res.reasons)
