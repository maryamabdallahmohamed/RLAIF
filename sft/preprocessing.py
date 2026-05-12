from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def get_tokenizer(model_name: str = "Qwen/Qwen2.5-0.5B") -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_smoltalk(split: str = "train") -> Dataset:
    return load_dataset("HuggingFaceTB/smoltalk", "all", split=split)


def tokenize_example(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_prompt_len: int,
    max_total_len: int,
) -> dict:
    messages = example["messages"]

    prompt_str = tokenizer.apply_chat_template(
        [messages[0]], tokenize=False, add_generation_prompt=True
    )

    full_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(
        prompt_str,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_len,
    )["input_ids"]

    full_ids = tokenizer(
        full_str,
        add_special_tokens=False,
        truncation=True,
        max_length=max_total_len,
    )["input_ids"]

    prompt_len = len(prompt_ids)

    input_ids = full_ids[:max_total_len]
    labels = [-100] * prompt_len + full_ids[prompt_len:max_total_len]

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
    num_proc: int = 4,
) -> Dataset:
    if dataset is None:
        dataset = load_smoltalk()

    dataset = dataset.filter(
        lambda x: len(x["messages"]) == 2,
        num_proc=num_proc,
        desc="Filtering 2-turn conversations",
    )

    max_total_len = max_prompt_len + max_response_len

    dataset = dataset.map(
        lambda x: tokenize_example(
            x,
            tokenizer,
            max_prompt_len,
            max_total_len,
        ),
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    dataset = dataset.filter(
        is_valid_sample,
        num_proc=num_proc,
    )

    return dataset
