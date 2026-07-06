# Agentic-search SFT — Stage 0 runbook

Distill GPT-5 general-web research traces into a Qwen2.5-7B-Instruct QLoRA adapter.
Same pipeline as the macro project, with `--domain general_search`. Budget: ~$50
teacher spend (~$2.50/trace measured at macro Stage 0), $0 training (Kaggle).

## 1. Data (Mac, costs OpenAI + Perplexity)

```bash
source .venv/bin/activate    # keys come from .env

python gen/make_questions.py  --domain general_search --n 25 --asof 2026-07-06 \
    --out search-fine-tuned-model/data/questions/bank.jsonl

python gen/generate_traces.py --domain general_search \
    --bank search-fine-tuned-model/data/questions/bank.jsonl \
    --out  search-fine-tuned-model/data/traces/raw \
    --limit 25 --model gpt-5 --concurrency 4

python gen/filter_traces.py   --domain general_search \
    --raw   search-fine-tuned-model/data/traces/raw \
    --clean search-fine-tuned-model/data/traces/clean \
    --judge gpt-5-mini

python gen/to_dataset.py \
    --clean search-fine-tuned-model/data/traces/clean \
    --out   search-fine-tuned-model/data/dataset
```

Expected yield: ~13–17 kept traces of 25. If fewer than 10 survive, read the
rejection histogram before spending more.

## 2. Train (Kaggle free GPU)

1. Zip the clean traces: `cd search-fine-tuned-model/data/traces && zip -r clean.zip clean/`.
2. Kaggle → Datasets → New Dataset → upload → name it `search-traces-clean` (private).
3. Open `train/kaggle_train_qlora.ipynb` on Kaggle. Config cell defaults already
   point at `search-traces-clean` / `Qwen/Qwen2.5-7B-Instruct` / `ashcash15/qwen2.5-7b-search`.
4. Add-ons → Secrets → `HF_TOKEN` = a Hugging Face **write** token.
5. Run all. The adapter lands in the Output tab AND at
   https://huggingface.co/ashcash15/qwen2.5-7b-search

## 3. Eval (GPU box or Kaggle)

```bash
python eval/make_eval_set.py --domain general_search --n 5 --asof 2026-07-06 \
    --out search-fine-tuned-model/data/eval_set.jsonl

python eval/run_eval_hf.py --domain general_search \
    --eval-set search-fine-tuned-model/data/eval_set.jsonl \
    --hf-model Qwen/Qwen2.5-7B-Instruct --adapter <path-or-repo-of-adapter> \
    --teacher gpt-5 --judge gpt-5-mini --limit 5 --max-steps 6
```

Write results to `docs/results/` (see `docs/results/2026-06-01-stage0.md` for the format).
