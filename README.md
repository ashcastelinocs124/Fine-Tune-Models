# Fine-Tune-Models — distilled agentic deep-research for global macro

Distill **frontier agent traces** into a small open-weight model (Qwen3-8B, QLoRA) that
performs **agentic deep research** for a global-macro fund: given a macro question it
plans, calls research tools, reads sources, and writes a cited findings report.

A teacher (GPT) runs a research loop over macro questions inside a **shared tool harness**;
the successful trajectories are filtered, converted to a loss-masked dataset, and used to
QLoRA-train the student. The **same harness** then drives the student at inference, which is
what makes a small distilled model actually transfer.

## Key ideas

- **One shared harness, two drivers.** The same `web_search` / `fetch_url` / `fred_series`
  tools, result formatting, and ReAct loop drive both teacher generation (OpenAI) and
  student serving (vLLM). Identical environments = the distillation transfers.
- **Rejection sampling.** Only traces that completed with a cited report, used the tools,
  and passed an LLM judge (≥4/5) enter the training set — a small model learns bad habits
  fast, so quality gating is load-bearing.
- **Visible reasoning (the hidden-CoT fix).** Frontier reasoning models don't expose their
  chain-of-thought over the API, so the teacher is prompted to write `Thought:` reasoning
  in the message stream before each tool call — making the reasoning a trainable target.

## Architecture

```mermaid
flowchart TB
  subgraph HARNESS["Shared tool harness - the spine"]
    direction LR
    ws["web_search via Sonar"]
    fu["fetch_url"]
    fr["fred_series"]
    loop["ReAct loop run_agent"]
  end

  subgraph GEN["1 Generate - teacher"]
    direction TB
    qb["Macro question bank"] --> oai["OpenAI driver - GPT runs the loop"] --> raw["Raw traces"]
  end

  subgraph FILTER["2 Filter - rejection sampling"]
    direction TB
    raw2["Raw traces"] --> gates["Mechanical gates"] --> judge["LLM judge keeps strong traces"] --> clean["Clean traces"]
  end

  subgraph DATA["3 Dataset"]
    direction TB
    clean2["Clean traces"] --> conv["to_dataset plus verified loss mask"] --> ds["train and val json"]
  end

  subgraph TRAIN["4 Train - VPS"]
    direction TB
    ds2["Dataset"] --> qlora["QLoRA via LLaMA-Factory"] --> adapter["Qwen3-8B LoRA adapter"]
  end

  subgraph SERVE["5 Serve and eval - VPS"]
    direction TB
    student["Student plus adapter"] --> vllm["vLLM driver runs the loop"] --> report["base vs student vs teacher report"]
  end

  GEN --> FILTER --> DATA --> TRAIN --> SERVE
  oai -.uses.-> loop
  vllm -.uses.-> loop
```

## Stack

| Piece | Choice |
|---|---|
| Student | Qwen3-8B, QLoRA (4-bit NF4) via LLaMA-Factory |
| Teacher | GPT (o-series / GPT-5) via OpenAI API |
| Tools | Perplexity Sonar (web search) · `fetch_url` · FRED (economic data) |
| Serving | vLLM (OpenAI-compatible) |
| Hardware | data pipeline on CPU/Mac · training + serving on 1× A100/H100 |

## Repo layout

```
macro_ds/   schema · formatting · tools · prompts · agent (ReAct loop) · drivers
            (OpenAI + vLLM) · questions · filtering · judge · dataset · mask_check · metrics
gen/        make_questions · generate_traces · filter_traces · to_dataset
train/      qlora_qwen3_8b.yaml · train.sh · check_template_parity.py   (run on VPS)
serve/      serve_vllm.sh · run_agent.py                                (run on VPS)
eval/       make_eval_set.py · run_eval.py                              (run on VPS)
tests/      unit tests for every deterministic component
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env     # OPENAI_API_KEY, PERPLEXITY_API_KEY, FRED_API_KEY, TEACHER_MODEL, JUDGE_MODEL, SONAR_MODEL
pytest -q                # the data pipeline + harness are fully unit-tested
```

The data pipeline (question bank → trace generation → filtering → dataset) runs on CPU.
Training (LLaMA-Factory QLoRA) and serving (vLLM) need a CUDA GPU:

```bash
pip install "llamafactory[torch,bitsandbytes,metrics]" vllm
```

## Usage

End-to-end commands are in **`RUNBOOK.md`** (generate → train → serve → eval), staged as a
cheap **Stage 0** smoke test (≈20 traces) to prove train/serve parity before the **Stage 1**
lean run (300–500 traces). Always run `python train/check_template_parity.py` and confirm
`PARITY OK` before training.

## Status

Prototype. The full pipeline is implemented and the harness + data pipeline are unit-tested;
training, serving, and evaluation scripts are written and run on the VPS.

## License

Proprietary — internal research tooling. All rights reserved.
