from macro_ds.questions import build_question_bank


def test_question_bank_is_unique_and_anchored():
    qs = build_question_bank(asof_date="2026-06-01", n=150)
    assert len(qs) == 150
    assert len({q["question"] for q in qs}) == 150  # unique
    assert all(q["asof_date"] == "2026-06-01" for q in qs)
    assert all("id" in q and "theme" in q for q in qs)


def test_question_bank_is_stable_across_calls():
    a = build_question_bank(asof_date="2026-06-01", n=50)
    b = build_question_bank(asof_date="2026-06-01", n=50)
    assert [q["id"] for q in a] == [q["id"] for q in b]  # deterministic order
    assert [q["question"] for q in a] == [q["question"] for q in b]


def test_question_bank_caps_at_available_combinations():
    # asking for more than the cross-product yields all unique combos, not duplicates
    qs = build_question_bank(asof_date="2026-06-01", n=100000)
    assert len(qs) == len({q["question"] for q in qs})


def test_eval_set_is_unique_and_disjoint_from_training():
    from macro_ds.questions import build_eval_set

    train = {q["question"] for q in build_question_bank(asof_date="2026-06-01", n=100000)}
    ev = build_eval_set(asof_date="2026-06-01", n=100000)
    ev_qs = {q["question"] for q in ev}
    assert len(ev_qs) == len(ev)  # unique
    assert train.isdisjoint(ev_qs)  # no leakage between train and eval
