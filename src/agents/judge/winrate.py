"""
Judge agent: pairwise win-rate scorer via Ollama Cloud.

Given a set of prompts and two model adapters (A and B), this module:
  1. Generates responses from each adapter for each prompt.
  2. Sends both orderings (A,B) and (B,A) to the Judge model.
  3. Parses the Judge's choice and computes A's win-rate with a bootstrap CI.

Position-bias mitigation: each (prompt, A_response, B_response) is judged twice,
once in each order. A win counts only if A is preferred in BOTH orderings
(strict) or averaged across orderings (default).

Usage:
    python -m src.agents.judge.winrate \\
        --adapter-a rlaif-ensemble --label-a "RLAIF-ensemble" \\
        --adapter-b rlaif-single   --label-b "RLAIF-single" \\
        --n-prompts 50 \\
        --output eval/winrate_ensemble_vs_single.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import torch
from datasets import load_dataset

from src.agents.critic.client import chat as ollama_chat
from src.utils.env import load_env

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

JUDGE_SYSTEM_PROMPT = """You are an impartial judge. You will be shown a user instruction and two candidate responses, labeled (A) and (B). Decide which response better follows the instruction — considering helpfulness, accuracy, and adherence to the requested format.

Reply with JSON only, no markdown: {"winner": "A" | "B" | "tie", "reason": "<one short sentence>"}"""

JUDGE_USER_TEMPLATE = """Instruction:
{prompt}

Response (A):
{response_a}

Response (B):
{response_b}

Which response is better?"""

WINNER_RE = re.compile(r'"winner"\s*:\s*"(A|B|tie)"', re.IGNORECASE)


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_policy(adapter_dir: str | None):
    """Load Qwen-0.5B with an optional LoRA adapter attached. None → base model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device()
    dtype = torch.float16 if device == "cuda" else None
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map={"": device} if device != "cpu" else None,
        attn_implementation="sdpa",
    )
    if adapter_dir is None:
        model = base
        tok_src = MODEL_NAME
    else:
        model = PeftModel.from_pretrained(base, adapter_dir).eval()
        tok_src = adapter_dir if (Path(adapter_dir) / "tokenizer_config.json").exists() else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    templated = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(templated, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # Strip the prompt-side prefix
    prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    if full.startswith(prompt_decoded):
        return full[len(prompt_decoded):].strip()
    return full.strip()


def _judge(prompt: str, response_a: str, response_b: str, judge_model: str, ollama_url: str) -> str:
    """Returns 'A', 'B', or 'tie' (lowercase 'tie')."""
    user = JUDGE_USER_TEMPLATE.format(prompt=prompt, response_a=response_a, response_b=response_b)
    reply = ollama_chat(ollama_url, judge_model, JUDGE_SYSTEM_PROMPT, user)
    m = WINNER_RE.search(reply)
    if not m:
        return "tie"
    w = m.group(1).upper()
    return w if w in ("A", "B") else "tie"


def _bootstrap_ci(wins: list[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    """Returns (mean, low, high) for the win-rate."""
    import numpy as np

    arr = np.array(wins, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(0)
    boots = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    boots.sort()
    low = boots[int(n_boot * alpha / 2)]
    high = boots[int(n_boot * (1 - alpha / 2))]
    return float(arr.mean()), float(low), float(high)


def run_winrate(
    adapter_a: str | None,
    adapter_b: str | None,
    label_a: str,
    label_b: str,
    n_prompts: int,
    judge_model: str,
    ollama_url: str,
    output_path: Path,
    seed: int = 0,
) -> dict:
    print(f"Loading held-out prompts (n={n_prompts})...")
    # Disjoint from DPO training (train[:2k]) and perturbation slice (train[60000:60100]).
    # UltraFeedback has ~63967 rows; tail slice is genuinely unseen.
    EVAL_START = 63000
    ds = load_dataset("openbmb/UltraFeedback", split=f"train[{EVAL_START}:{EVAL_START + n_prompts}]")
    prompts = [s["instruction"] for s in ds if isinstance(s.get("instruction"), str) and s["instruction"].strip()]

    print(f"Generating responses from A ({label_a})...")
    model_a, tok_a = _load_policy(adapter_a)
    responses_a = [_generate(model_a, tok_a, p) for p in prompts]
    del model_a
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"Generating responses from B ({label_b})...")
    model_b, tok_b = _load_policy(adapter_b)
    responses_b = [_generate(model_b, tok_b, p) for p in prompts]
    del model_b

    print(f"Judging {len(prompts)} pairs × 2 orderings = {2 * len(prompts)} judge calls...")
    rng = random.Random(seed)
    records: list[dict] = []
    a_wins: list[float] = []  # fractional: 1.0 if A wins both, 0.5 if split, 0.0 if B wins both
    for i, (p, ra, rb) in enumerate(zip(prompts, responses_a, responses_b)):
        # Ordering 1: (A, B)
        w1 = _judge(p, ra, rb, judge_model, ollama_url)
        # Ordering 2: (B, A) — A is now the second response
        w2 = _judge(p, rb, ra, judge_model, ollama_url)
        a_wins_ord1 = 1 if w1 == "A" else 0 if w1 == "B" else 0.5
        a_wins_ord2 = 1 if w2 == "B" else 0 if w2 == "A" else 0.5  # flipped
        a_score = (a_wins_ord1 + a_wins_ord2) / 2.0
        a_wins.append(a_score)
        records.append({
            "idx": i,
            "prompt": p,
            "response_a": ra,
            "response_b": rb,
            "ordering1_winner": w1,
            "ordering2_winner": w2,
            "a_score": a_score,
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(prompts)} judged. Running A win-rate: {sum(a_wins)/len(a_wins):.3f}")

    mean, low, high = _bootstrap_ci(a_wins)
    summary = {
        "label_a": label_a,
        "label_b": label_b,
        "adapter_a": adapter_a,
        "adapter_b": adapter_b,
        "n_prompts": len(prompts),
        "judge_model": judge_model,
        "a_winrate_mean": mean,
        "a_winrate_ci95": [low, high],
        "n_decisive_a": int(sum(1 for s in a_wins if s == 1.0)),
        "n_decisive_b": int(sum(1 for s in a_wins if s == 0.0)),
        "n_split": int(sum(1 for s in a_wins if 0 < s < 1.0)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(f"\n=== {label_a} vs {label_b} ===")
    print(f"  A win-rate: {mean:.3f}  [95% CI: {low:.3f}, {high:.3f}]")
    print(f"  Decisive A: {summary['n_decisive_a']}  Decisive B: {summary['n_decisive_b']}  Split: {summary['n_split']}")
    print(f"  Saved: {output_path}")
    return summary


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-a", default=None, help="LoRA adapter dir for A (None = base Qwen)")
    parser.add_argument("--adapter-b", default=None, help="LoRA adapter dir for B (None = base Qwen)")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--n-prompts", type=int, default=50)
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", "gpt-oss:120b-cloud"))
    parser.add_argument("--ollama-url", default=os.environ.get("OLLAMA_URL", "https://ollama.com"))
    parser.add_argument("--output", required=True, help="Output JSON path.")
    args = parser.parse_args()

    run_winrate(
        adapter_a=args.adapter_a,
        adapter_b=args.adapter_b,
        label_a=args.label_a,
        label_b=args.label_b,
        n_prompts=args.n_prompts,
        judge_model=args.judge_model,
        ollama_url=args.ollama_url,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
