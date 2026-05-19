"""
Precise-RLAIF training: DPO loss applied to synthetic (original, Critic-perturbed) pairs.

Loads a starting LoRA adapter (typically the DPO checkpoint), continues training it with
TRL's DPOTrainer where:
    chosen   = original_response       (un-perturbed)
    rejected = perturbed_response      (one sentence degraded by the Critic agent)

The dataset path is a JSONL produced by scripts/generate_perturbation_datasets.py.

Usage:
    # Single-Critic variant (combined constitution):
    python -m src.agents.generator.precise_rlaif \\
        --dataset data/perturbation/combined.jsonl \\
        --output-dir rlaif-single \\
        --base-adapter dpo-qwen-0.5b

    # Ensemble variant (equal-mix of all 4 constitutions):
    python -m src.agents.generator.precise_rlaif \\
        --dataset data/perturbation/ensemble_equal_mix.jsonl \\
        --output-dir rlaif-ensemble \\
        --base-adapter dpo-qwen-0.5b
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from sft.preprocessing import get_tokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# DPO-on-perturbations hyperparams (mirror the DPO baseline)
BETA = 0.2
LR = 5e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 16  # effective batch 16
NUM_EPOCHS = 1
MAX_LENGTH = 512
MAX_PROMPT_LEN = 256
LOGGING_STEPS = 10


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_perturbation_dataset(path: Path, tokenizer, max_pairs: int | None = None):
    """JSONL → HF Dataset of {prompt (chat-templated), chosen, rejected}."""
    from datasets import Dataset

    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chosen = rec.get("original_response")
            rejected = rec.get("perturbed_response")
            prompt = rec.get("prompt")
            if not (chosen and rejected and prompt):
                continue
            if chosen.strip() == rejected.strip():
                continue
            templated = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            rows.append({"prompt": templated, "chosen": chosen, "rejected": rejected})
            if max_pairs is not None and len(rows) >= max_pairs:
                break
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precise-RLAIF (DPO on Critic perturbations).")
    parser.add_argument("--dataset", required=True, help="Path to perturbation JSONL.")
    parser.add_argument("--output-dir", required=True, help="Where to save the resulting adapter.")
    parser.add_argument("--base-adapter", default="dpo-qwen-0.5b",
                        help="Starting LoRA adapter (default: DPO checkpoint).")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--fast", action="store_true", help="Smoke mode: 200 pairs, 20 steps.")
    args = parser.parse_args()

    if args.fast:
        args.max_pairs = 200
        args.max_steps = 20

    device = _get_device()
    print(f"Device: {device}")
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = device == "cuda" and not use_bf16
    print(f"Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")

    base_adapter = Path(args.base_adapter)
    tokenizer_source = (
        str(base_adapter) if (base_adapter / "tokenizer_config.json").exists() else MODEL_NAME
    )
    print(f"Tokenizer from: {tokenizer_source}")
    tokenizer = get_tokenizer(tokenizer_source)

    print(f"Loading base {MODEL_NAME}...")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else None,
        device_map={"": device} if device != "cpu" else None,
        attn_implementation="sdpa",
    )
    base.enable_input_require_grads()

    print(f"Attaching starting adapter from {args.base_adapter}...")
    model = PeftModel.from_pretrained(base, args.base_adapter, is_trainable=True)
    model.print_trainable_parameters()

    print(f"Loading perturbation dataset from {args.dataset}...")
    dataset = load_perturbation_dataset(Path(args.dataset), tokenizer, max_pairs=args.max_pairs)
    print(f"Dataset size: {len(dataset):,}")
    if len(dataset) == 0:
        raise SystemExit("No usable pairs found in dataset.")

    dpo_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=NUM_EPOCHS,
        max_steps=args.max_steps,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=GRAD_CLIP,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=LOGGING_STEPS,
        save_strategy="no" if args.fast else "epoch",
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=(device == "cuda"),
        remove_unused_columns=False,
        optim="adamw_torch",
        beta=BETA,
        max_length=MAX_LENGTH,
    )
    accepted = set(inspect.signature(DPOConfig.__init__).parameters)
    if "max_prompt_length" in accepted:
        dpo_kwargs["max_prompt_length"] = MAX_PROMPT_LEN
    if "save_strategy" not in accepted:
        dpo_kwargs.pop("save_strategy", None)

    dpo_config = DPOConfig(**dpo_kwargs)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # uses base + adapter-disabled as frozen reference
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting Precise-RLAIF training...")
    trainer.train()

    if not args.fast:
        print(f"Saving adapter to {args.output_dir}/")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
