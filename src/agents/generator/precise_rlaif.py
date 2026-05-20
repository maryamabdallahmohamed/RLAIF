"""
Precise-RLAIF: DPO loss on (original, Critic-perturbed) pairs.

chosen   = original_response
rejected = perturbed_response  (one sentence degraded by Critic)
"""
import inspect
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
from datasets import Dataset

from sft.preprocessing import get_tokenizer

# ---- config -----------------------------------------------------------
DATASET = "data/perturbation/combined.jsonl"    
OUTPUT_DIR = "rlaif-single"                       
BASE_ADAPTER = "dpo-qwen-0.5b"

FAST = False               
MAX_PAIRS: int | None = None
MAX_STEPS = -1

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
BETA = 0.2
LR = 5e-6
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PER_DEVICE_BATCH = 1
GRAD_ACCUM_STEPS = 16        
NUM_EPOCHS = 1
MAX_LENGTH = 512
MAX_PROMPT_LEN = 256
LOGGING_STEPS = 10
# -----------------------------------------------------------------------

if FAST:
    MAX_PAIRS, MAX_STEPS = 200, 20


def load_perturbation_dataset(path: Path, tokenizer, max_pairs: int | None = None):
    """JSONL → HF Dataset of {prompt (chat-templated), chosen, rejected}."""
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chosen, rejected, prompt = (
                rec.get("original_response"),
                rec.get("perturbed_response"),
                rec.get("prompt"),
            )
            if not (chosen and rejected and prompt) or chosen.strip() == rejected.strip():
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


# ---- setup ------------------------------------------------------------
device = "mps"

use_bf16 = device == "mps" or (device == "cuda" and torch.cuda.is_bf16_supported())
use_fp16 = device == "mps" or (device == "cuda" and torch.cuda.is_fp16_supported())
print(f"Device: {device} | Precision: {'bf16' if use_bf16 else 'fp16' if use_fp16 else 'fp32'}")

base_adapter_path = Path(BASE_ADAPTER)
tokenizer_source = (
    str(base_adapter_path) if (base_adapter_path / "tokenizer_config.json").exists() else MODEL_NAME
)
tokenizer = get_tokenizer(tokenizer_source)

# ---- model ------------------------------------------------------------
print(f"Loading base {MODEL_NAME}...")
base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else None,
    device_map={"": device} if device != "cpu" else None,
    attn_implementation="sdpa",
)
base.enable_input_require_grads()

print(f"Attaching starting adapter from {BASE_ADAPTER}...")
model = PeftModel.from_pretrained(base, BASE_ADAPTER, is_trainable=True)
model.print_trainable_parameters()

# ---- data -------------------------------------------------------------
print(f"Loading perturbation dataset from {DATASET}...")
dataset = load_perturbation_dataset(Path(DATASET), tokenizer, max_pairs=MAX_PAIRS)
print(f"Dataset size: {len(dataset):,}")
if len(dataset) == 0:
    raise SystemExit("No usable pairs found in dataset.")

# ---- trainer ----------------------------------------------------------
dpo_kwargs = dict(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=GRAD_CLIP,
    bf16=use_bf16,
    fp16=use_fp16,
    logging_steps=LOGGING_STEPS,
    save_strategy="no" if FAST else "epoch",
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

trainer = DPOTrainer(
    model=model,
    ref_model=None,               
    args=DPOConfig(**dpo_kwargs),
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ---- train ------------------------------------------------------------
print("Starting Precise-RLAIF training...")
trainer.train()

if not FAST:
    print(f"Saving adapter to {OUTPUT_DIR}/")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
print("Done.")