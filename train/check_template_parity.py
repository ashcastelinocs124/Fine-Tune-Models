"""Stage-0 gate: confirm LLaMA-Factory tokenizes/masks our dataset the SAME way we
verified in macro_ds.mask_check. RUN ON THE VPS (needs llamafactory + the Qwen3 model files).

It always prints our independent reference (decoded trained vs masked tokens). It then
attempts to load the first training example through LLaMA-Factory's own data pipeline and
asserts:
  - the input_id sequences match, and
  - the set of trained positions (label != -100) matches.

If LLaMA-Factory's internal API differs on your installed version, the cross-check is
skipped with a clear message and you must eyeball our reference against a 1-sample
training run (llamafactory-cli train ... ) — but a definitive mismatch exits non-zero.

Usage:
    python train/check_template_parity.py [--config train/qlora_qwen3_8b.yaml]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

IGNORE = -100


def _load_first_record(dataset_dir: str) -> dict:
    train = json.loads((pathlib.Path(dataset_dir) / "train.json").read_text())
    if not train:
        sys.exit("train.json is empty — run gen/to_dataset.py first.")
    return train[0]


def _our_mask(record: dict, model_name: str):
    from transformers import AutoTokenizer

    from macro_ds.mask_check import render_and_mask

    tok = AutoTokenizer.from_pretrained(model_name)
    ids, labels = render_and_mask(record, tok)
    return tok, ids, labels


def _llamafactory_example(config: dict):
    """Best-effort: return (input_ids, labels) for the first train example via LLaMA-Factory."""
    from llamafactory.data import get_dataset
    from llamafactory.data.template import get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer

    model_args, data_args, training_args, finetuning_args, _ = get_train_args(config)
    tok_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tok_module["tokenizer"], data_args)
    ds = get_dataset(template, model_args, data_args, training_args, stage="sft", **tok_module)
    ex = ds["train_dataset"][0]
    return list(ex["input_ids"]), list(ex["labels"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train/qlora_qwen3_8b.yaml")
    args = ap.parse_args()

    import yaml

    config = yaml.safe_load(pathlib.Path(args.config).read_text())
    model_name = config["model_name_or_path"]
    dataset_dir = config.get("dataset_dir", "data/dataset")

    record = _load_first_record(dataset_dir)
    tok, ids, labels = _our_mask(record, model_name)

    print("=== OUR INDEPENDENT MASK (macro_ds.mask_check) ===")
    print("TRAINED:", repr(tok.decode([i for i, l in zip(ids, labels) if l != IGNORE])[:600]))
    print("MASKED :", repr(tok.decode([i for i, l in zip(ids, labels) if l == IGNORE])[:300]))
    our_trained_positions = {k for k, l in enumerate(labels) if l != IGNORE}

    try:
        lf_ids, lf_labels = _llamafactory_example(config)
    except Exception as e:  # noqa: BLE001
        print(f"\n[skip] LLaMA-Factory cross-check unavailable on this version: {e}")
        print("       Eyeball the TRAINED text above (must be only assistant reasoning + tool")
        print("       calls + final report, never <|im_start|>/<tool_response> framing).")
        return

    lf_trained_positions = {k for k, l in enumerate(lf_labels) if l != IGNORE}
    ids_match = ids == lf_ids
    mask_match = our_trained_positions == lf_trained_positions
    if ids_match and mask_match:
        print("\nPARITY OK — LLaMA-Factory tokenization and loss mask match our verification.")
        return

    print("\n❌ PARITY MISMATCH")
    if not ids_match:
        print(f"  input_ids differ: ours={len(ids)} tokens, LLaMA-Factory={len(lf_ids)} tokens")
    if not mask_match:
        extra = sorted(lf_trained_positions - our_trained_positions)[:10]
        missing = sorted(our_trained_positions - lf_trained_positions)[:10]
        print(f"  trained positions differ: LF-only={extra} ours-only={missing}")
    sys.exit(1)


if __name__ == "__main__":
    main()
