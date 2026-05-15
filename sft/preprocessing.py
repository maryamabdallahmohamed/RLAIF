from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_tokenizer(model_name: str = "Qwen/Qwen2.5-0.5B") -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_smoltalk(split: str = "train", max_samples: int | None = None) -> Dataset:
    slice_suffix = "[:50000]" if split == "train" else ""
    dataset = load_dataset("HuggingFaceTB/smoltalk", "all", split=f"{split}{slice_suffix}")

    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    return dataset


def tokenize_example(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_total_len: int,
) -> dict:
    messages = example["messages"]

    prompt_str = tokenizer.apply_chat_template(
        [messages[0]], tokenize=False, add_generation_prompt=True
    )

    full_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    full_ids = tokenizer(
        full_str,
        add_special_tokens=False,
        truncation=True,
        max_length=max_total_len,
    )["input_ids"]

    # Measure the prompt boundary inside the full string so the mask offset
    # is exact — avoids the off-by-one that arises from tokenizing prompt_str
    # standalone (special tokens / BOS interact differently at string boundaries).
    prompt_len = len(tokenizer(
        full_str[:len(prompt_str)],
        add_special_tokens=False,
    )["input_ids"])
    prompt_len = min(prompt_len, len(full_ids))

    input_ids = full_ids
    labels = [-100] * prompt_len + full_ids[prompt_len:]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def is_valid_sample(example: dict) -> bool:
    return len(example["input_ids"]) > 10


def build_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset | None = None,
    max_prompt_len: int = 256,
    max_response_len: int = 1024,
    num_proc: int = 2,
    max_samples: int | None = None,
) -> Dataset:
    if dataset is None:
        dataset = load_smoltalk(max_samples=max_samples)

    if max_samples is not None and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    dataset = dataset.filter(
        lambda x: len(x["messages"]) == 2,
        num_proc=num_proc,
        desc="Filtering 2-turn conversations",
    )

    max_total_len = max_prompt_len + max_response_len

    dataset = dataset.map(
        lambda x: tokenize_example(x, tokenizer, max_total_len),
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    dataset = dataset.filter(
        is_valid_sample,
        num_proc=num_proc,
    )

    return dataset
