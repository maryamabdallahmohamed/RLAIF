"""
Ultrafeedback → DPO triples (prompt, chosen, rejected).

For each instruction we keep the completion with the highest `overall_score`
as `chosen` and the lowest as `rejected`. Pairs with a score gap below
`min_score_gap` are dropped to filter out near-ties.

TRL's DPOTrainer tokenizes; we hand it raw strings with the Qwen chat
template applied to the prompt.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase

ULTRAFEEDBACK_PATH = "openbmb/UltraFeedback"


def _completion_score(completion: dict) -> float | None:
    """Best-effort numeric score for a single UltraFeedback completion.

    Tries `overall_score` first; falls back to averaging the four
    annotation dimensions when only those are present.
    """
    score = completion.get("overall_score")
    if score is not None:
        try:
            return float(score)
        except (TypeError, ValueError):
            pass

    annotations = completion.get("annotations") or {}
    ratings: list[float] = []
    for dim in ("instruction_following", "helpfulness", "honesty", "truthfulness"):
        dim_obj = annotations.get(dim)
        if not isinstance(dim_obj, dict):
            continue
        raw = dim_obj.get("Rating")
        try:
            ratings.append(float(raw))
        except (TypeError, ValueError):
            continue
    if ratings:
        return sum(ratings) / len(ratings)
    return None


def _extract_pair(sample: dict, min_score_gap: float) -> dict | None:
    """Pick best/worst completion for one Ultrafeedback row."""
    completions = sample.get("completions") or []
    scored: list[tuple[float, str]] = []
    for c in completions:
        s = _completion_score(c)
        if s is None:
            continue
        response = c.get("response")
        if not isinstance(response, str) or not response.strip():
            continue
        scored.append((s, response))

    if len(scored) < 2:
        return None

    scored.sort(key=lambda x: x[0])
    worst_score, worst_resp = scored[0]
    best_score, best_resp = scored[-1]
    if best_score - worst_score < min_score_gap:
        return None
    if best_resp.strip() == worst_resp.strip():
        return None

    prompt = sample.get("instruction")
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    return {
        "prompt": prompt,
        "chosen": best_resp,
        "rejected": worst_resp,
        "score_gap": best_score - worst_score,
    }


def load_ultrafeedback_pairs(
    max_pairs: int = 10_000,
    min_score_gap: float = 0.5,
    split_slice: str = "train",
    seed: int = 42,
) -> "Dataset":
    """Load raw Ultrafeedback rows and convert to (prompt, chosen, rejected)."""
    from datasets import Dataset, load_dataset

    raw = load_dataset(ULTRAFEEDBACK_PATH, split=split_slice)
    pairs: list[dict] = []
    for sample in raw:
        pair = _extract_pair(sample, min_score_gap=min_score_gap)
        if pair is not None:
            pairs.append(pair)
        if len(pairs) >= max_pairs:
            break

    ds = Dataset.from_list(pairs)
    return ds.shuffle(seed=seed)


def apply_chat_template_to_prompts(
    dataset: "Dataset",
    tokenizer: "PreTrainedTokenizerBase",
    num_proc: int = 2,
) -> "Dataset":
    """Wrap each prompt with the Qwen chat template (assistant turn open)."""

    def _map(example: dict) -> dict:
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"prompt": templated}

    return dataset.map(_map, num_proc=num_proc, desc="Applying Qwen chat template")


def build_dpo_dataset(
    tokenizer: "PreTrainedTokenizerBase | None" = None,
    max_pairs: int = 10_000,
    min_score_gap: float = 0.5,
    num_proc: int = 2,
) -> "Dataset":
    """One-shot loader used by the training script."""
    if tokenizer is None:
        from sft.preprocessing import get_tokenizer  # reuse Qwen tokenizer + pad-token handling
        tokenizer = get_tokenizer()
    raw_pairs = load_ultrafeedback_pairs(max_pairs=max_pairs, min_score_gap=min_score_gap)
    return apply_chat_template_to_prompts(raw_pairs, tokenizer, num_proc=num_proc)
