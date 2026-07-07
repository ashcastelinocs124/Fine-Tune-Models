# Runbook — generate → train → serve → eval

End-to-end commands. `[mac]` = local/dev (no GPU); `[vps]` = A100/H100 box.
Design: `docs/plans/2026-06-01-macro-deepsearch-distill-design.md`. Plan: `…-plan.md`.

## 0. Setup

`[mac]` (data pipeline + tests; no GPU):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # fill OPENAI_API_KEY, FRED_API_KEY, TEACHER_MODEL, JUDGE_MODEL, SEARCH_MODEL
pytest -q              # 53 tests (one downloads the Qwen3 tokenizer)
```

`[vps]` (training + serving; needs CUDA):
```bash
git clone <repo> && cd fine-tune-models
pip install -e .
pip install "llamafactory[torch,bitsandbytes,metrics]"   # or clone LLaMA-Factory + pip install -e .
pip install vllm
cp .env.example .env   # same keys (teacher + judge run via OpenAI API)
nvidia-smi             # confirm the GPU
```

## 1. Data (run on `[mac]`, costs OpenAI only — search runs via the OpenAI web_search tool)

```bash
python gen/make_questions.py --n 180 --asof 2026-06-01           # data/questions/bank.jsonl
# STAGE 0 smoke test: ~20 traces first (≈$50) to prove the pipeline end-to-end
python gen/generate_traces.py --bank data/questions/bank.jsonl --out data/traces/raw \
    --limit 20 --model "$TEACHER_MODEL" --concurrency 4
python gen/filter_traces.py  --raw data/traces/raw --clean data/traces/clean --judge "$JUDGE_MODEL"
python gen/to_dataset.py     --clean data/traces/clean --out data/dataset
```
Eyeball a raw trace: confirm visible "Thought:" reasoning precedes each tool call.

## 2. Train (`[vps]`)

```bash
# register the dataset: merge data/dataset/dataset_info.json's "macro_traces" into
# LLaMA-Factory's dataset_info.json, OR rely on dataset_dir: data/dataset in the yaml.
python train/check_template_parity.py          # MUST print "PARITY OK" (Stage-0 gate)
bash train/train.sh                             # QLoRA -> outputs/qwen3-8b-macro-lora
```
If parity fails, STOP and fix `gen/to_dataset.py` / the template mapping before training
(see learnings.md for the Qwen3 masking gotchas).

## 3. Serve (`[vps]`)

```bash
bash serve/serve_vllm.sh    # OpenAI-compatible endpoint on :8000 ("macro" = student)
# in another shell:
python serve/run_agent.py --question "Path of the US 10y yield next quarter?" --asof 2026-06-01
```

## 4. Eval (`[vps]`)

```bash
python eval/make_eval_set.py --n 48 --asof 2026-06-01            # held-out, disjoint from training
python eval/run_eval.py --eval-set eval/eval_set.jsonl \
    --teacher "$TEACHER_MODEL" --judge "$JUDGE_MODEL" --limit 30 --out eval/results
```
Prints a base-vs-student-vs-teacher table. Success = the student closes most of the
base→teacher gap on valid_tool_call_rate, completion, groundedness, and judge_overall.

## 5. Scale up (Stage 1)

Once Stage 0 is clean: regenerate with the full bank (drop `--limit`, optionally 1–2
samples/question to reach 300–500 filtered traces), retrain (2 epochs), re-eval, and write
up `docs/results/`. Track OpenAI spend against the ~$1–1.5k ceiling.
