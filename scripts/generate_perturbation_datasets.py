"""
Full perturbation dataset generation (Member B task, steps 36-42).

Produces 6 JSONL files under data/perturbation/:
  instruction_following.jsonl  (5k pairs)
  helpfulness.jsonl            (5k pairs)
  truthfulness.jsonl           (5k pairs)
  combined.jsonl               (5k pairs)  ← single-Critic baseline
  ensemble_equal_mix.jsonl     (5k pairs, 1.25k from each constitution)
  ensemble_agreement_weighted.jsonl  (5k pairs, higher weight when all 4 critics agree on sentence)

Schema per record:
  {prompt, original_response, perturbed_sentence_idx, perturbed_response, constitution[, weight]}

Usage:
    python scripts/generate_perturbation_datasets.py [--model llama3.1:8b] [--url http://localhost:11434]
    python scripts/generate_perturbation_datasets.py --held-out-start 50000 --n-pairs 1000 --n-perturbations 5
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

from src.agents.critic.agent import CriticAgent, CriticParseError

CONSTITUTIONS = ["instruction_following", "helpfulness", "truthfulness", "combined"]
N_PAIRS = 1000
N_PERTURBATIONS = 5
# Held-out slice starts well past any smoke-test / Member-A training slice.
HELD_OUT_START = 50000
OUTPUT_DIR = Path("data/perturbation")
CACHE_DIR = Path("data/critic_cache")
CONSTITUTIONS_PATH = Path("src/agents/critic/constitutions.yaml")

SCHEMA_FIELDS = {"prompt", "original_response", "perturbed_sentence_idx", "perturbed_response", "constitution"}


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_record(record: dict) -> list[str]:
    errors: list[str] = []
    missing = SCHEMA_FIELDS - record.keys()
    if missing:
        errors.append(f"missing fields: {missing}")
    if "prompt" in record and not isinstance(record["prompt"], str):
        errors.append("prompt must be str")
    if "original_response" in record and not isinstance(record["original_response"], str):
        errors.append("original_response must be str")
    if "perturbed_response" in record and not isinstance(record["perturbed_response"], str):
        errors.append("perturbed_response must be str")
    if "perturbed_sentence_idx" in record and not isinstance(record["perturbed_sentence_idx"], int):
        errors.append("perturbed_sentence_idx must be int")
    if "constitution" in record and record["constitution"] not in CONSTITUTIONS:
        errors.append(f"unknown constitution: {record['constitution']!r}")
    return errors


def validate_file(path: Path) -> int:
    """Returns number of validation errors found."""
    errors_total = 0
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [INVALID JSON] line {lineno}: {e}")
                errors_total += 1
                continue
            errs = validate_record(record)
            for err in errs:
                print(f"  [SCHEMA ERROR] line {lineno}: {err}")
                errors_total += len(errs)
    return errors_total


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_pairs(held_out_start: int, n_pairs: int) -> list[tuple[str, str]]:
    end = held_out_start + n_pairs
    print(f"Loading UltraFeedback held-out slice train[{held_out_start}:{end}]...")
    ds = load_dataset("openbmb/UltraFeedback", split=f"train[{held_out_start}:{end}]")
    pairs = []
    for sample in ds:
        prompt: str = sample["instruction"]
        response: str = sample["completions"][0]["response"]
        if prompt.strip() and response.strip():
            pairs.append((prompt, response))
    print(f"  Loaded {len(pairs)} valid pairs.")
    return pairs


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_constitution_dataset(
    critic: CriticAgent,
    pairs: list[tuple[str, str]],
    constitution: str,
    n_perturbations: int,
    output_path: Path,
) -> list[dict]:
    records: list[dict] = []
    parse_errors = 0
    total = len(pairs) * n_perturbations

    print(f"\n[{constitution}] Generating {total} perturbations ({len(pairs)} pairs × {n_perturbations})...")

    with open(output_path, "w") as f:
        for i, (prompt, response) in enumerate(pairs):
            for pidx in range(n_perturbations):
                try:
                    perturbed, sentence_idx = critic.perturb(
                        prompt, response, constitution, perturbation_idx=pidx
                    )
                    record = {
                        "prompt": prompt,
                        "original_response": response,
                        "perturbed_sentence_idx": sentence_idx,
                        "perturbed_response": perturbed,
                        "constitution": constitution,
                    }
                    records.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                except CriticParseError as e:
                    parse_errors += 1
                    print(f"  [PARSE ERROR] pair {i}, perturbation {pidx}: {e}")

            if (i + 1) % 100 == 0:
                done = (i + 1) * n_perturbations
                print(f"  {i + 1}/{len(pairs)} pairs ({done}/{total} perturbations, {parse_errors} errors so far)")

    print(f"  Done. {len(records)}/{total} written, {parse_errors} parse errors. → {output_path}")
    return records


# ---------------------------------------------------------------------------
# Ensemble construction
# ---------------------------------------------------------------------------

def build_equal_mix(
    datasets: dict[str, list[dict]],
    output_path: Path,
    target_total: int = 5000,
) -> None:
    per_constitution = target_total // len(datasets)
    records: list[dict] = []
    for constitution, dataset in datasets.items():
        sample = random.sample(dataset, min(per_constitution, len(dataset)))
        records.extend(sample)
    random.shuffle(records)
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n[ensemble_equal_mix] {len(records)} records → {output_path}")


def build_agreement_weighted(
    datasets: dict[str, list[dict]],
    output_path: Path,
    target_total: int = 5000,
    high_weight: float = 2.0,
    low_weight: float = 1.0,
) -> None:
    """
    Pairs where all 4 constitutions agree on the same sentence_idx get high_weight.
    Agreement is measured per (prompt, original_response) pair, using the modal
    sentence_idx across all perturbations from all constitutions.
    """
    # Group sentence_idx values by (prompt, response)
    pair_to_idxs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for dataset in datasets.values():
        for r in dataset:
            key = (r["prompt"], r["original_response"])
            pair_to_idxs[key].append(r["perturbed_sentence_idx"])

    # A pair has "full agreement" if all 4 constitutions' modal idx is the same
    def modal(values: list[int]) -> int:
        from collections import Counter
        return Counter(values).most_common(1)[0][0]

    # We need per-constitution modal idx for each pair
    pair_constitution_modal: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for constitution, dataset in datasets.items():
        by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
        for r in dataset:
            key = (r["prompt"], r["original_response"])
            by_pair[key].append(r["perturbed_sentence_idx"])
        for key, idxs in by_pair.items():
            pair_constitution_modal[key][constitution] = modal(idxs)

    high_agreement_keys: set[tuple[str, str]] = set()
    for key, const_modal in pair_constitution_modal.items():
        if len(const_modal) == len(datasets):
            unique_modals = set(const_modal.values())
            if len(unique_modals) == 1:
                high_agreement_keys.add(key)

    # Build equal-mix base then assign weights
    per_constitution = target_total // len(datasets)
    records: list[dict] = []
    for constitution, dataset in datasets.items():
        sample = random.sample(dataset, min(per_constitution, len(dataset)))
        records.extend(sample)

    high_agreement_count = 0
    with open(output_path, "w") as f:
        for r in records:
            key = (r["prompt"], r["original_response"])
            weight = high_weight if key in high_agreement_keys else low_weight
            if weight == high_weight:
                high_agreement_count += 1
            out = dict(r)
            out["weight"] = weight
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(
        f"\n[ensemble_agreement_weighted] {len(records)} records "
        f"({high_agreement_count} high-weight, {len(records) - high_agreement_count} low-weight) "
        f"→ {output_path}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full perturbation datasets (Member B).")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--held-out-start", type=int, default=HELD_OUT_START,
                        help="Start index of held-out UltraFeedback slice (default: 50000)")
    parser.add_argument("--n-pairs", type=int, default=N_PAIRS)
    parser.add_argument("--n-perturbations", type=int, default=N_PERTURBATIONS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.held_out_start, args.n_pairs)
    if len(pairs) < args.n_pairs:
        print(f"WARNING: only {len(pairs)} pairs available (requested {args.n_pairs}).")

    critic = CriticAgent(
        ollama_url=args.url,
        model=args.model,
        constitutions_path=CONSTITUTIONS_PATH,
        cache_dir=CACHE_DIR,
    )

    all_datasets: dict[str, list[dict]] = {}
    for constitution in CONSTITUTIONS:
        output_path = OUTPUT_DIR / f"{constitution}.jsonl"
        records = generate_constitution_dataset(
            critic, pairs, constitution, args.n_perturbations, output_path
        )
        all_datasets[constitution] = records

    build_equal_mix(all_datasets, OUTPUT_DIR / "ensemble_equal_mix.jsonl")
    build_agreement_weighted(all_datasets, OUTPUT_DIR / "ensemble_agreement_weighted.jsonl")

    # Schema validation (step 42)
    print("\n--- Schema validation ---")
    all_clean = True
    for path in sorted(OUTPUT_DIR.glob("*.jsonl")):
        n_errors = validate_file(path)
        status = "OK" if n_errors == 0 else f"{n_errors} ERRORS"
        print(f"  {path.name}: {status}")
        if n_errors:
            all_clean = False

    if not all_clean:
        print("\nSchema validation FAILED. Fix errors before handoff.")
        sys.exit(1)

    print("\nAll datasets validated. Ready for handoff.")


if __name__ == "__main__":
    main()
