"""Macro research question bank.

Deterministic cross-product of macro THEMES × question TEMPLATES. The model learns
the research *process*, so we want broad, realistic analyst questions across the macro
landscape (rates, FX, commodities, growth, credit), each anchored to an as-of date.
"""

from __future__ import annotations

from typing import Any

# Each theme supplies the fields the templates interpolate.
THEMES: list[dict[str, str]] = [
    {"key": "fed", "subject": "the Fed's policy rate", "asset": "front-end USD rates",
     "trade": "being long 2y Treasuries", "data": "US CPI and payrolls"},
    {"key": "ecb", "subject": "the ECB's policy rate", "asset": "Bunds",
     "trade": "long Bunds vs Treasuries", "data": "euro-area HICP"},
    {"key": "boj", "subject": "the BoJ's policy normalization", "asset": "JGBs and the yen",
     "trade": "short JGBs", "data": "Japan CPI and wage growth"},
    {"key": "us_infl", "subject": "US inflation", "asset": "TIPS breakevens",
     "trade": "long inflation breakevens", "data": "US CPI and PCE"},
    {"key": "us_recession", "subject": "US recession risk", "asset": "US equities",
     "trade": "long duration as a recession hedge", "data": "US labor-market and ISM"},
    {"key": "dxy", "subject": "the US dollar (DXY)", "asset": "the dollar index",
     "trade": "short USD versus a basket", "data": "US growth and rate-differential"},
    {"key": "eurusd", "subject": "EUR/USD", "asset": "EUR/USD",
     "trade": "long EUR/USD", "data": "euro-area versus US growth"},
    {"key": "usdjpy", "subject": "USD/JPY", "asset": "the yen",
     "trade": "the JPY carry trade", "data": "US-Japan rate-differential"},
    {"key": "curve", "subject": "the US yield curve (2s10s)", "asset": "the Treasury curve",
     "trade": "a 2s10s steepener", "data": "Treasury auction and inflation"},
    {"key": "oil", "subject": "crude oil prices", "asset": "WTI crude",
     "trade": "long crude oil", "data": "inventory and OPEC supply"},
    {"key": "gold", "subject": "gold prices", "asset": "gold",
     "trade": "long gold", "data": "real-yield and central-bank-buying"},
    {"key": "copper", "subject": "copper prices", "asset": "copper",
     "trade": "long copper", "data": "China activity and inventory"},
    {"key": "china", "subject": "China's growth trajectory", "asset": "China-sensitive assets",
     "trade": "long China growth proxies", "data": "China PMI and credit"},
    {"key": "em_debt", "subject": "emerging-market sovereign debt", "asset": "EM hard-currency debt",
     "trade": "long EM sovereign credit", "data": "EM inflation and the dollar"},
    {"key": "us_fiscal", "subject": "the US fiscal trajectory", "asset": "the long end of the curve",
     "trade": "long-end steepeners on heavy supply", "data": "Treasury issuance and the deficit"},
    {"key": "credit", "subject": "US credit spreads", "asset": "US investment-grade credit",
     "trade": "long IG credit", "data": "default-rate and growth"},
]

# Templates reference {subject}, {asset}, {trade}, {data}.
TEMPLATES: list[str] = [
    "What is the most likely path of {subject} over the next quarter, and why?",
    "What is the most likely path of {subject} over the next 12 months, and why?",
    "Lay out the bull and bear case for {trade}.",
    "What do recent {data} releases imply for {asset}?",
    "What are the key macro risks to {asset} over the next two quarters?",
    "How should a global-macro fund position around {subject} right now?",
    "What is the market currently pricing for {subject}, and is it mispriced?",
    "Summarize the latest data and central-bank signals shaping {subject}.",
    "What catalysts over the next quarter could move {asset} sharply in either direction?",
    "Build a base, bull, and bear scenario tree for {subject}.",
]


# Eval-only templates — deliberately DISJOINT from TEMPLATES so the held-out eval set
# never overlaps the training questions (honest generalization measurement).
EVAL_TEMPLATES: list[str] = [
    "Assess the balance of risks around {subject} heading into the next policy window.",
    "If you had to put on one position expressing a view on {subject} today, what would it be and why?",
    "Which single upcoming data release matters most for {asset}, and what would each outcome imply?",
]


def cross_product(
    asof_date: str, themes: list[dict[str, str]], templates: list[str], id_prefix: str, n: int
) -> list[dict[str, Any]]:
    """Deterministic themes × templates question bank (shared by all domain profiles)."""
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for theme in themes:
        for ti, template in enumerate(templates):
            question = template.format(**theme)
            if question in seen:
                continue
            seen.add(question)
            items.append(
                {
                    "id": f"{id_prefix}{theme['key']}-{ti:02d}",
                    "question": question,
                    "theme": theme["key"],
                    "asof_date": asof_date,
                }
            )
    return items[:n]


def build_question_bank(asof_date: str, n: int) -> list[dict[str, Any]]:
    """Return up to `n` unique, deterministically-ordered macro TRAINING questions."""
    return cross_product(asof_date, THEMES, TEMPLATES, "", n)


def build_eval_set(asof_date: str, n: int) -> list[dict[str, Any]]:
    """Return up to `n` held-out EVAL questions, disjoint from the training bank."""
    return cross_product(asof_date, THEMES, EVAL_TEMPLATES, "eval-", n)
