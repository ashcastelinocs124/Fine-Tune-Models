"""Direct QLoRA trainer for Qwen3-8B on the macro traces. RUN ON THE VPS (needs GPU).

Why not LLaMA-Factory: our traces use visible-ReAct assistant turns (a "Thought:" plus a
tool_call in the SAME turn). LLaMA-Factory's sharegpt tool format represents tool calls as
separate `function_call` turns that carry only the call JSON, dropping the reasoning — the
exact signal we distill. Instead we tokenize each trace with `macro_ds.mask_check.render_and_mask`
(verified against the real Qwen3 template: only assistant reasoning + tool calls + final report
get loss) and train a LoRA with a plain Trainer. Train/serve parity is guaranteed because the
SAME chat template renders training data here and the prompt at vLLM inference time.

Usage:
    python train/train_qlora.py --clean data/traces/clean --out outputs/qwen3-8b-macro-lora \
        --epochs 2 --max-len 16384
"""

from __future__ import annotations

import argparse
import glob

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from macro_ds.dataset import trace_to_record
from macro_ds.mask_check import render_and_mask
from macro_ds.schema import Trace

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_examples(clean_dir: str, tok, max_len: int) -> list[dict]:
    examples = []
    for f in sorted(glob.glob(f"{clean_dir}/*.json")):
        trace = Trace.from_json(open(f).read())
        try:
            ids, labels = render_and_mask(trace_to_record(trace), tok)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {f}: {e}")
            continue
        ids, labels = ids[:max_len], labels[:max_len]
        if not any(l != -100 for l in labels):
            print(f"[skip] {f}: no trainable tokens after truncation")
            continue
        examples.append({"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)})
    return examples


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append(b["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--clean", default="data/traces/clean")
    ap.add_argument("--out", default="outputs/qwen3-8b-macro-lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    examples = build_examples(args.clean, tok, args.max_len)
    if not examples:
        raise SystemExit("no training examples — generate + filter traces first")
    print(f"Training on {len(examples)} examples; "
          f"trainable tokens: {[sum(1 for l in e['labels'] if l != -100) for e in examples]}")
    # eyeball the first example's trained span
    e0 = examples[0]
    trained = tok.decode([i for i, l in zip(e0["input_ids"], e0["labels"]) if l != -100])
    print("=== first example TRAINED text (first 300 chars) ===")
    print(trained[:300])

    # Pre-Ampere GPUs (Kaggle T4/P100) have no bf16 — fall back to fp16.
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, torch_dtype=dtype, device_map={"": 0}
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=LORA_TARGETS, task_type="CAUSAL_LM", bias="none",
        ),
    )
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=bf16,
        fp16=not bf16,
        logging_steps=1,
        save_strategy="no",  # save once explicitly at the end (avoids mid-train checkpoint disk use)
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=Dataset.from_list(examples),
        data_collator=PadCollator(tok.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved LoRA adapter -> {args.out}")


if __name__ == "__main__":
    main()
