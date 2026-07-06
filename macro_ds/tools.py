"""The three research tools, shared by teacher generation and student serving.

Design rules:
- Tool errors are returned as `"ERROR: ..."` strings, never raised. The agent must
  SEE failures (in the tool result) and recover — exactly as it will at inference.
- Each public tool delegates network I/O to a small `_raw` helper that is easy to mock
  in tests and carries tenacity retries for transient failures.
- Output is rendered by the deterministic functions in `formatting.py`.
"""

from __future__ import annotations

from typing import Any, Callable

import httpx
import trafilatura
from tenacity import retry, stop_after_attempt, wait_exponential

from macro_ds import config
from macro_ds.formatting import format_fetched_page, format_fred, format_search_results

FRED_BASE = "https://api.stlouisfed.org/fred"
SONAR_ENDPOINT = "https://api.perplexity.ai/chat/completions"  # OpenAI-compatible
_HTTP_TIMEOUT = 20.0
_SONAR_TIMEOUT = 60.0  # Sonar searches the web + synthesizes, so it's slower
_USER_AGENT = "macro-ds/0.1 (research agent)"

_retry = retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8), reraise=True)


# ---------------------------------------------------------------------------
# Raw network helpers (mocked in tests)
# ---------------------------------------------------------------------------
@_retry
def _sonar_raw(query: str, k: int) -> list[dict[str, Any]]:
    """Web search via Perplexity Sonar. Returns ranked {title, url, snippet} hits.

    Sonar's response carries a top-level `search_results` array (title/url/snippet/date);
    we map it to the same shape Tavily used so `format_search_results` is unchanged. Older
    API responses expose only `citations` (plain URLs) — handled as a fallback.
    """
    model = config.get_opt("SONAR_MODEL", "sonar")
    resp = httpx.post(
        SONAR_ENDPOINT,
        headers={
            "Authorization": f"Bearer {config.get_key('PERPLEXITY_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={"model": model, "messages": [{"role": "user", "content": query}], "search_mode": "web"},
        timeout=_SONAR_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    results = [
        {"title": r.get("title"), "url": r.get("url"), "snippet": r.get("snippet")}
        for r in (data.get("search_results") or [])
    ]
    if not results:  # fallback for responses that only return citation URLs
        results = [{"title": u, "url": u, "snippet": ""} for u in (data.get("citations") or [])]
    return results[:k]


@_retry
def _http_get(url: str) -> str:
    resp = httpx.get(
        url, timeout=_HTTP_TIMEOUT, follow_redirects=True, headers={"User-Agent": _USER_AGENT}
    )
    resp.raise_for_status()
    return resp.text


@_retry
def _fred_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    params = {**params, "api_key": config.get_key("FRED_API_KEY"), "file_type": "json"}
    resp = httpx.get(f"{FRED_BASE}/{path}", params=params, timeout=_HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _fred_raw(series_id: str, start: str | None, end: str | None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"series_id": series_id}
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end
    data = _fred_get("series/observations", params)
    return data.get("observations", [])


def _fred_search_raw(text: str) -> list[dict[str, Any]]:
    data = _fred_get("series/search", {"search_text": text, "limit": 10})
    return data.get("seriess", [])


# ---------------------------------------------------------------------------
# Public tools (return strings; errors are data, not exceptions)
# ---------------------------------------------------------------------------
def web_search(query: str, k: int = 5) -> str:
    try:
        return format_search_results(_sonar_raw(query, k), k=k)
    except Exception as e:  # noqa: BLE001 — surface failure as tool output
        return f"ERROR: web_search failed: {e}"


def fetch_url(url: str, max_chars: int = 6000) -> str:
    try:
        html = _http_get(url)
        text = trafilatura.extract(html) or ""
        if not text.strip():
            return f"ERROR: fetch_url got no extractable text from {url}"
        return format_fetched_page(url, text, max_chars=max_chars)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: fetch_url failed for {url}: {e}"


def fred_series(series_id: str, start: str | None = None, end: str | None = None) -> str:
    try:
        return format_fred(series_id, _fred_raw(series_id, start, end))
    except Exception as e:  # noqa: BLE001
        return f"ERROR: fred_series failed for {series_id}: {e}"


def fred_search(text: str) -> str:
    try:
        rows = _fred_search_raw(text)
        if not rows:
            return f"FRED search for {text!r}: no matching series."
        lines = [f"  {r.get('id')}: {r.get('title')}" for r in rows]
        return f"FRED series matching {text!r}:\n" + "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"ERROR: fred_search failed for {text!r}: {e}"


# ---------------------------------------------------------------------------
# Registry + OpenAI tool schemas
# ---------------------------------------------------------------------------
def _schema(name: str, description: str, params: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": params, "required": required},
        },
    }


TOOLS: dict[str, tuple[Callable[..., str], dict[str, Any]]] = {
    "web_search": (
        web_search,
        _schema(
            "web_search",
            "Search the open web for relevant pages. Returns a ranked list of title/url/snippet.",
            {
                "query": {"type": "string", "description": "search query"},
                "k": {"type": "integer", "description": "number of results (default 5)"},
            },
            ["query"],
        ),
    ),
    "fetch_url": (
        fetch_url,
        _schema(
            "fetch_url",
            "Fetch a URL and return its main readable text (truncated). Use to read a source in full.",
            {"url": {"type": "string", "description": "the page URL to read"}},
            ["url"],
        ),
    ),
    "fred_series": (
        fred_series,
        _schema(
            "fred_series",
            "Fetch observations for a FRED economic series (e.g. DGS10, CPIAUCSL, DEXUSEU, DCOILWTICO).",
            {
                "series_id": {"type": "string", "description": "FRED series id"},
                "start": {"type": "string", "description": "YYYY-MM-DD observation start (optional)"},
                "end": {"type": "string", "description": "YYYY-MM-DD observation end (optional)"},
            },
            ["series_id"],
        ),
    ),
    "fred_search": (
        fred_search,
        _schema(
            "fred_search",
            "Search FRED for series ids matching a text query when you don't know the series id.",
            {"text": {"type": "string", "description": "free-text description of the data you want"}},
            ["text"],
        ),
    ),
}


def openai_tool_schemas(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Return tool schemas in OpenAI tools format. `names` selects a subset (None = all)."""
    if names is None:
        return [schema for _, schema in TOOLS.values()]
    unknown = [n for n in names if n not in TOOLS]
    if unknown:
        raise ValueError(f"unknown tool(s) {unknown!r}; valid tools: {list(TOOLS)}")
    return [TOOLS[n][1] for n in names]


def execute_tool(name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a tool call by name; unknown tool / bad args become ERROR strings."""
    entry = TOOLS.get(name)
    if entry is None:
        return f"ERROR: unknown tool {name!r}"
    fn = entry[0]
    try:
        return fn(**arguments)
    except TypeError as e:
        return f"ERROR: bad arguments for {name}: {e}"
