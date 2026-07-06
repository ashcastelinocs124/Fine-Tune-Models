"""Domain profiles: the five domain-specific knobs of the distillation pipeline.

Everything else (agent loop, drivers, filtering gates, dataset build, training) is
domain-agnostic. A profile bundles: system prompt, finalize instruction, judge rubric,
allowed tools, and the themes x templates question bank. `MACRO` reproduces the
original pipeline exactly; `GENERAL_SEARCH` is the any-topic web-research domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from macro_ds import judge as _judge
from macro_ds import prompts as _prompts
from macro_ds import questions as _questions


@dataclass(frozen=True)
class DomainProfile:
    name: str
    system_template: str
    finalize_instruction: str
    judge_rubric: str
    tool_names: list[str] | None  # None = all registered tools
    themes: list[dict[str, str]]
    templates: list[str]
    eval_templates: list[str]

    def build_system_prompt(self, asof_date: str, max_steps: int = 12) -> str:
        return self.system_template.format(asof_date=asof_date, max_steps=max_steps)

    def build_question_bank(self, asof_date: str, n: int) -> list[dict[str, Any]]:
        return _questions.cross_product(asof_date, self.themes, self.templates, "", n)

    def build_eval_set(self, asof_date: str, n: int) -> list[dict[str, Any]]:
        return _questions.cross_product(asof_date, self.themes, self.eval_templates, "eval-", n)


# ---------------------------------------------------------------------------
# macro — the original pipeline, byte-identical
# ---------------------------------------------------------------------------
MACRO = DomainProfile(
    name="macro",
    system_template=_prompts.SYSTEM_TEMPLATE,
    finalize_instruction=_prompts.FINALIZE_INSTRUCTION,
    judge_rubric=_judge.RUBRIC,
    tool_names=None,
    themes=_questions.THEMES,
    templates=_questions.TEMPLATES,
    eval_templates=_questions.EVAL_TEMPLATES,
)


# ---------------------------------------------------------------------------
# general_search — any-topic deep web research with citations
# ---------------------------------------------------------------------------
GENERAL_SYSTEM_TEMPLATE = """You are a research analyst doing deep, source-grounded research on any topic.
Today's date (as-of) is {asof_date}. Treat this as "now" for all reasoning.

You have these tools:
- web_search(query): find relevant pages on the open web.
- fetch_url(url): read the full text of a specific page.

Method — follow this loop:
1. Before EVERY tool call, write a short reasoning line that starts with "Thought:"
   explaining what you are about to check and why. The Thought goes in your message
   content; the tool call follows it.
2. Search broadly, then fetch the most relevant sources to read them in full.
3. Prefer primary sources (papers, official reports, first-party data) over
   commentary when the facts matter.
4. Iterate: refine queries, chase down the key open questions, resolve contradictions
   between sources.

Grounding rules:
- Ground every factual claim in something you actually retrieved. Do NOT invent
  numbers, dates, or quotes.
- Cite sources inline: page URLs in [brackets].

When you are confident you can answer, stop calling tools and write the final
deliverable as a structured report (begin it with "Final report:"). The report should
state the answer, the key findings each with supporting evidence and citations, the
remaining uncertainties, and what further evidence would settle them.

You have a budget of about {max_steps} tool-calling steps — be efficient and decisive.
"""

GENERAL_FINALIZE_INSTRUCTION = (
    "You have reached your research budget — do NOT call any more tools. "
    "Write your Final report now, beginning with 'Final report:'. Structure it as: the answer, "
    "the key findings each with an inline citation (source URLs in [brackets]), the remaining "
    "uncertainties, and what further evidence would settle them. Use only what you have already "
    "retrieved; do not invent numbers."
)

GENERAL_RUBRIC = """You are a strict reviewer of deep-research traces.
Score the analyst's FINAL REPORT against the SOURCES THAT WERE ACTUALLY RETRIEVED.

Rate each 1-5 (5 best):
- groundedness: every claim is supported by retrieved sources; no invented numbers.
- coverage: addresses the question's key aspects and counter-arguments.
- citation_faithfulness: cited URLs actually support the cited claims.
- structure: clear answer, findings, evidence, uncertainties, and what would settle them.

Respond ONLY with a JSON object:
{"groundedness": int, "coverage": int, "citation_faithfulness": int,
 "structure": int, "overall": number, "rationale": string}
"""

# Each theme supplies {subject}, {debate}, {comparison}, {data} for the templates.
GENERAL_THEMES: list[dict[str, str]] = [
    {"key": "llm", "subject": "large language models",
     "debate": "whether AI progress is hitting a scaling wall",
     "comparison": "open-weight versus closed frontier AI models",
     "data": "AI benchmark and enterprise-adoption"},
    {"key": "climate", "subject": "global decarbonization",
     "debate": "whether solar plus storage can displace baseload fossil power",
     "comparison": "battery-electric versus hydrogen for heavy transport",
     "data": "global emissions and renewable-capacity"},
    {"key": "glp1", "subject": "GLP-1 weight-loss drugs",
     "debate": "whether GLP-1 drugs meaningfully reduce long-term cardiovascular risk",
     "comparison": "GLP-1 drugs versus bariatric surgery",
     "data": "clinical-trial and prescription-volume"},
    {"key": "space", "subject": "the commercial space-launch industry",
     "debate": "whether fully reusable rockets will cut launch costs by an order of magnitude",
     "comparison": "commercial launch providers versus national space agencies",
     "data": "launch-cadence and cost-per-kilogram"},
    {"key": "semis", "subject": "the semiconductor supply chain",
     "debate": "whether advanced-node chip manufacturing can diversify beyond Taiwan",
     "comparison": "TSMC's versus Intel's foundry roadmap",
     "data": "fab capital-expenditure and chip-export"},
    {"key": "nuclear", "subject": "nuclear power's revival",
     "debate": "whether small modular reactors are economically viable",
     "comparison": "new nuclear versus renewables-plus-storage",
     "data": "electricity-cost and reactor-construction-timeline"},
    {"key": "longevity", "subject": "the science of aging and longevity",
     "debate": "whether any current intervention meaningfully extends healthy human lifespan",
     "comparison": "caloric-restriction mimetics versus senolytics",
     "data": "lifespan-trial and biomarker"},
    {"key": "edu", "subject": "AI's impact on education",
     "debate": "whether AI tutors improve learning outcomes at scale",
     "comparison": "AI tutoring versus traditional small-group instruction",
     "data": "learning-outcome and adoption"},
    {"key": "quantum", "subject": "quantum computing",
     "debate": "whether quantum computers will reach commercially useful advantage this decade",
     "comparison": "superconducting versus trapped-ion qubit platforms",
     "data": "qubit-count and error-rate"},
    {"key": "crispr", "subject": "CRISPR gene-editing therapies",
     "debate": "whether in-vivo gene editing is ready for common diseases",
     "comparison": "base editing versus traditional CRISPR-Cas9",
     "data": "clinical-trial and approval"},
    {"key": "housing", "subject": "housing affordability in major cities",
     "debate": "whether upzoning measurably lowers rents",
     "comparison": "land-value taxes versus inclusionary zoning",
     "data": "housing-permit and rent"},
    {"key": "evs", "subject": "the electric-vehicle transition",
     "debate": "whether EV adoption keeps accelerating without subsidies",
     "comparison": "Chinese versus Western EV manufacturers",
     "data": "EV-sales and battery-cost"},
]

GENERAL_TEMPLATES: list[str] = [
    "What is the current state of {subject}, and where is it heading over the next few years?",
    "{debate} — lay out the strongest evidence on both sides and give your assessment.",
    "Compare {comparison}: which is better positioned today, and why?",
    "What do the latest {data} figures tell us about {subject}?",
    "What are the biggest risks and open questions around {subject}?",
    "Write a research brief on {subject} for a smart non-expert audience.",
    "What recent developments have most changed expert thinking on {subject}?",
    "Build a best-case, base-case, and worst-case scenario for {subject} over the next five years.",
]

# Deliberately DISJOINT from GENERAL_TEMPLATES so eval never overlaps training.
GENERAL_EVAL_TEMPLATES: list[str] = [
    "Assess the balance of evidence on {debate} as of today.",
    "What single upcoming development would most change the outlook for {subject}, and what would each outcome imply?",
    "Which widely-made claims about {subject} are best supported by evidence, and which are hype?",
]

GENERAL_SEARCH = DomainProfile(
    name="general_search",
    system_template=GENERAL_SYSTEM_TEMPLATE,
    finalize_instruction=GENERAL_FINALIZE_INSTRUCTION,
    judge_rubric=GENERAL_RUBRIC,
    tool_names=["web_search", "fetch_url"],
    themes=GENERAL_THEMES,
    templates=GENERAL_TEMPLATES,
    eval_templates=GENERAL_EVAL_TEMPLATES,
)


_PROFILES: dict[str, DomainProfile] = {p.name: p for p in (MACRO, GENERAL_SEARCH)}


def get_profile(name: str) -> DomainProfile:
    """Resolve a profile by name; unknown names fail fast listing the valid ones."""
    profile = _PROFILES.get(name)
    if profile is None:
        raise ValueError(f"unknown domain {name!r}; valid domains: {sorted(_PROFILES)}")
    return profile
