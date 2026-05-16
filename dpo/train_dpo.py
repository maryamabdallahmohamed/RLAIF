"""
DPO training for Qwen 2.5 0.5B starting from the SFT LoRA adapter.

Loads the SFT adapter as the starting policy, continues training it with
TRL's DPOTrainer against preference pairs from openbmb/UltraFeedback.
`ref_model=None` tells TRL to use the base model with the LoRA disabled
as the frozen reference policy — only one copy of the base lives in VRAM.

Usage:
    python -m dpo.train_dpo            # full run
    python -m dpo.train_dpo --fast     # smoke run (~20 steps)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

from dpo.preprocessing import build_dpo_dataset
from sft.preprocessing import get_tokenizer

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
SFT_ADAPTER_DIR = "sft-qwen-0.5b"
OUTPUT_DIR = "checkpoints/dpo-qwen-0.5b"

# Dataset
MAX_PAIRS = 5_000
MIN_SCORE_GAP = 0.5

# DPO
BETA = 0.2
LR = 5e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PER_DEVICE_BATCH = 2
GRAD_ACCUM_STEPS = 8  # effective batch = 16
NUM_EPOCHS = 1
MAX_LENGTH = 512       # was 1024; halving length is the biggest T4 speedup
MAX_PROMPT_LEN = 256   # was 512
LOGGING_STEPS = 25

# Fast mode (smoke test) — just verify it runs and loss moves; 20 steps is enough
FAST_MAX_PAIRS = 200
FAST_MAX_STEPS = 20

# LoRA (matches the SFT adapter so the layers line up cleanly)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ── Device ────────────────────────────────────────────────────────────────────

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training on Qwen 2.5 0.5B SFT.")
    parser.add_argument("--fast", action="store_true",
                        help="Smoke mode: ~200 pairs, 20 steps.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--sft-adapter", default=SFT_ADAPTER_DIR)
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Override MAX_PAIRS (full=10000, fast=500).")
    args = parser.parse_args()

    fast = args.fast
    if args.max_pairs is None:
        max_pairs = FAST_MAX_PAIRS if fast else MAX_PAIRS
    else:
        max_pairs = args.max_pairs

    device = _get_device()
    print(f"Device: {device}")
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        cap_major, cap_minor = torch.cuda.get_device_capability(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} (sm_{cap_major}{cap_minor})")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    # Prefer the tokenizer saved alongside the SFT adapter so the chat template
    # is byte-for-byte identical to what produced the warm-start.
    sft_dir = Path(args.sft_adapter)
    tokenizer_source = str(sft_dir) if (sft_dir / "tokenizer_config.json").exists() else MODEL_NAME
    print(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = get_tokenizer(tokenizer_source)

    # ── Model: base → load SFT LoRA → keep as the policy starting point ─────
    print(f"Loading base model {MODEL_NAME}...")
    # Only use bf16 when the GPU actually supports it natively (Ampere+, sm_80+).
    # On Turing (T4, sm_75), bf16 is emulated and ~2x slower than fp16.
    use_bf16 = device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8
    use_fp16 = device == "cuda" and not use_bf16
    print(f"Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else None,
        device_map={"": device} if device != "cpu" else None,
        attn_implementation="sdpa",
    )
    base.enable_input_require_grads()

    print(f"Attaching SFT LoRA adapter from {args.sft_adapter}...")
    model = PeftModel.from_pretrained(
        base,
        args.sft_adapter,
        is_trainable=True,  # we'll train this adapter further with DPO
    )
    # No gradient_checkpointing — 0.5B in fp16 fits in T4's 15 GB easily.
    model.print_trainable_parameters()

    # ── Dataset ──────────────────────────────────────────────────────────────
    print(f"Building DPO dataset (max_pairs={max_pairs})...")
    dataset = build_dpo_dataset(
        tokenizer=tokenizer,
        max_pairs=max_pairs,
        min_score_gap=MIN_SCORE_GAP,
    )
    print(f"Dataset size: {len(dataset):,}")

    # ── DPO config ───────────────────────────────────────────────────────────
    # Build kwargs dict and filter against the installed DPOConfig signature
    # so the script survives TRL renaming fields between versions.
    import inspect

    dpo_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=NUM_EPOCHS,
        max_steps=FAST_MAX_STEPS if fast else -1,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=GRAD_CLIP,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10 if fast else LOGGING_STEPS,
        save_strategy="no" if fast else "epoch",
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
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
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    if not fast:
        print(f"Saving DPO adapter to {args.output_dir}/")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
