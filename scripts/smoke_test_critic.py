"""
Smoke test for CriticAgent against UltraFeedback.

Usage (Ollama Cloud, default — reads OLLAMA_API_KEY/CRITIC_MODEL/OLLAMA_URL from .env):
    python scripts/smoke_test_critic.py --n 5

Usage (local Ollama):
    python scripts/smoke_test_critic.py --model llama3.1:8b --url http://localhost:11434

Writes results to data/smoke_test_<N>.jsonl.
"""
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset

from src.agents.critic.agent import CriticAgent, CriticParseError
from src.agents.critic.segment import segment
from src.utils.env import load_env

CONSTITUTIONS = ["instruction_following", "helpfulness", "truthfulness", "combined"]
CACHE_DIR = Path("data/critic_cache")
CONSTITUTIONS_PATH = Path("src/agents/critic/constitutions.yaml")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("CRITIC_MODEL", "gpt-oss:20b-cloud"))
    parser.add_argument("--url", default=os.environ.get("OLLAMA_URL", "https://ollama.com"))
    parser.add_argument("--n", type=int, default=5, help="Number of prompts")
    args = parser.parse_args()
    N_SAMPLES = args.n
    OUTPUT_PATH = Path(f"data/smoke_test_{N_SAMPLES}.jsonl")

    print(f"Critic model: {args.model} @ {args.url}")
    print(f"Loading UltraFeedback (first {N_SAMPLES} samples)...")
    ds = load_dataset("openbmb/UltraFeedback", split=f"train[:{N_SAMPLES}]")

    critic = CriticAgent(
        ollama_url=args.url,
        model=args.model,
        constitutions_path=CONSTITUTIONS_PATH,
        cache_dir=CACHE_DIR,
    )

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    total = 0
    parse_errors = 0
    out_of_range = 0

    with open(OUTPUT_PATH, "w") as f:
        for i, sample in enumerate(ds):
            prompt: str = sample["instruction"]
            chosen: str = sample["completions"][0]["response"]

            for constitution_name in CONSTITUTIONS:
                total += 1
                try:
                    rejected, sentence_idx = critic.perturb(prompt, chosen, constitution_name)
                    sentences = segment(chosen)
                    if not (0 <= sentence_idx < len(sentences)):
                        out_of_range += 1
                    f.write(
                        json.dumps({
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": rejected,
                            "constitution": constitution_name,
                            "sentence_idx": sentence_idx,
                        }) + "\n"
                    )
                except CriticParseError as e:
                    parse_errors += 1
                    print(f"  [PARSE ERROR] sample {i}, {constitution_name}: {e}")
                except Exception as e:
                    parse_errors += 1
                    print(f"  [NETWORK ERROR] sample {i}, {constitution_name}: {type(e).__name__}: {e}")

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{N_SAMPLES} samples processed...")

    successful = total - parse_errors
    print(f"\n--- Smoke Test Summary ---")
    print(f"Total pairs attempted : {total}")
    print(f"Parse errors          : {parse_errors}/{total} ({100 * parse_errors / total:.1f}%)")
    print(f"Out-of-range idx      : {out_of_range}/{successful} ({100 * out_of_range / max(1, successful):.1f}%)")
    print(f"Output                : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
