import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from sft.preprocessing import build_sft_dataset, get_tokenizer, load_smoltalk

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "sft-qwen-0.5b"

MAX_PROMPT_LEN = 256
MAX_RESPONSE_LEN = 1024

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

LR = 1e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
PER_DEVICE_BATCH = 2
GRAD_ACCUM_STEPS = 16  # effective batch = 32
NUM_EPOCHS = 1
LOGGING_STEPS = 50

# ── Device ────────────────────────────────────────────────────────────────────

def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = _get_device()
    print(f"Device: {device}")

    tokenizer = get_tokenizer(MODEL_NAME)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Building dataset...")
    raw = load_smoltalk("train")
    dataset = build_sft_dataset(
        tokenizer=tokenizer,
        dataset=raw,
        max_prompt_len=MAX_PROMPT_LEN,
        max_response_len=MAX_RESPONSE_LEN,
    )
    print(f"Dataset size after filtering: {len(dataset):,}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=GRAD_CLIP,
        bf16=True,
        fp16=False,
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=0,  
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving checkpoint to {OUTPUT_DIR}/")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
