"""Deterministic tool-result formatting.

These functions turn raw tool outputs into the exact strings the model sees as tool
results. They MUST be pure and deterministic — same input → byte-identical output —
because the teacher (generation) and student (serving) both rely on them. Any
non-determinism here (timestamps, dict ordering, randomness) silently breaks parity.
"""

from __future__ import annotations

from typing import Any

SNIPPET_CHARS = 240
PAGE_DEFAULT_CHARS = 6000


def _truncate(text: str, n: int) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "…"


def format_search_results(results: list[dict[str, Any]], k: int = 5) -> str:
    """Render web-search hits as a stable numbered list."""
    if not results:
        return "No results found."
    lines = []
    for i, r in enumerate(results[:k], start=1):
        title = (r.get("title") or "(untitled)").strip()
        url = (r.get("url") or "").strip()
        snippet = _truncate(r.get("snippet") or r.get("content") or "", SNIPPET_CHARS)
        lines.append(f"[{i}] {title}\n    {url}\n    {snippet}")
    return "\n".join(lines)


def format_fred(series_id: str, observations: list[dict[str, Any]], max_rows: int = 24) -> str:
    """Render FRED observations as a compact, row-capped table.

    Keeps the most recent `max_rows` observations (FRED returns oldest-first).
    """
    if not observations:
        return f"FRED {series_id}: no observations."
    rows = observations[-max_rows:]
    body = "\n".join(f"  {o.get('date')}  {o.get('value')}" for o in rows)
    return f"FRED series {series_id} (latest {len(rows)} obs):\n{body}"


def format_fetched_page(url: str, text: str, max_chars: int = PAGE_DEFAULT_CHARS) -> str:
    """Render fetched page text with a header and hard truncation."""
    body = (text or "").strip()
    suffix = ""
    if len(body) > max_chars:
        body = body[:max_chars].rstrip()
        suffix = f"\n[truncated to {max_chars} chars]"
    return f"PAGE {url}\n{body}{suffix}"
