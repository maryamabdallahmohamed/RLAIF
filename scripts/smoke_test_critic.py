"""
100-pair smoke test for CriticAgent against UltraFeedback.

Usage:
    python scripts/smoke_test_critic.py [--model llama3.1:8b] [--url http://localhost:11434]

Writes results to data/smoke_test_100.jsonl.
Prints: total pairs, parse error rate, sentence_idx out-of-range rate.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset

from src.agents.critic.agent import CriticAgent, CriticParseError
from src.agents.critic.segment import segment

CONSTITUTIONS = ["instruction_following", "helpfulness", "truthfulness", "combined"]
N_SAMPLES = 100
OUTPUT_PATH = Path("data/smoke_test_100.jsonl")
CACHE_DIR = Path("data/critic_cache")
CONSTITUTIONS_PATH = Path("src/agents/critic/constitutions.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--url", default="http://localhost:11434")
    args = parser.parse_args()

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
