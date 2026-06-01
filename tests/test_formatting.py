from macro_ds.formatting import (
    SNIPPET_CHARS,
    format_fetched_page,
    format_fred,
    format_search_results,
)


def test_search_formatting_is_deterministic_and_capped():
    results = [
        {"title": f"T{i}", "url": f"http://x/{i}", "snippet": "s" * 500} for i in range(8)
    ]
    out1 = format_search_results(results, k=5)
    out2 = format_search_results(results, k=5)
    assert out1 == out2  # deterministic — byte identical
    assert out1.count("http://") == 5  # capped to k
    # each snippet truncated to ~SNIPPET_CHARS, so total stays bounded
    assert len(out1) < 5 * (SNIPPET_CHARS + 200)


def test_search_formatting_handles_empty():
    assert "No results" in format_search_results([], k=5)


def test_search_formatting_truncates_long_snippet():
    out = format_search_results([{"title": "T", "url": "u", "snippet": "x" * 1000}], k=5)
    assert "x" * (SNIPPET_CHARS + 1) not in out  # truncated
    assert "…" in out  # truncation marker


def test_fred_formatting_is_tabular_and_row_capped():
    obs = [{"date": f"2026-01-{d:02d}", "value": str(d)} for d in range(1, 40)]
    out = format_fred("DGS10", obs, max_rows=10)
    assert "DGS10" in out
    assert out.count("2026-01-") == 10  # capped rows
    # deterministic
    assert out == format_fred("DGS10", obs, max_rows=10)


def test_fetched_page_truncates_to_max_chars():
    text = "word " * 5000
    out = format_fetched_page("http://u", text, max_chars=200)
    assert "http://u" in out
    assert "[truncated" in out
    # body portion must not exceed max_chars (header excluded by checking the marker presence)
    assert len(out) < 200 + 200  # body cap + small header/footer
