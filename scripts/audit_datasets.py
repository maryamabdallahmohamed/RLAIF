"""
Manual quality audit for perturbation datasets (step 43).

Pulls 20 random examples per constitution, prints them for human review,
and checks the automated invariant: exactly one sentence was modified
and the modified sentence matches the constitution's intent (via string diff).

Usage:
    python scripts/audit_datasets.py [--input-dir data/perturbation] [--n 20] [--seed 42]
"""
import argparse
import json
import random
from pathlib import Path

from src.agents.critic.segment import segment

INPUT_DIR = Path("data/perturbation")
CONSTITUTIONS = ["instruction_following", "helpfulness", "truthfulness", "combined"]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def count_changed_sentences(original: str, perturbed: str) -> tuple[int, list[int]]:
    """Returns (n_changed, list_of_changed_indices)."""
    orig_sents = segment(original)
    pert_sents = segment(perturbed)
    changed = [
        i for i, (o, p) in enumerate(zip(orig_sents, pert_sents)) if o != p
    ]
    # Also flag length mismatch (added/removed sentences)
    if len(orig_sents) != len(pert_sents):
        changed.append(-1)  # sentinel for structural change
    return len(changed), changed


def audit_constitution(path: Path, n: int, seed: int) -> dict:
    records = load_jsonl(path)
    sample = random.Random(seed).sample(records, min(n, len(records)))

    exactly_one = 0
    idx_matches = 0
    structural_changes = 0

    print(f"\n{'='*70}")
    print(f"Constitution: {path.stem}  ({len(records)} total records, auditing {len(sample)})")
    print(f"{'='*70}")

    for i, r in enumerate(sample):
        orig = r["original_response"]
        pert = r["perturbed_response"]
        claimed_idx = r["perturbed_sentence_idx"]
        n_changed, changed_idxs = count_changed_sentences(orig, pert)

        one_sentence_changed = n_changed == 1 and -1 not in changed_idxs
        if one_sentence_changed:
            exactly_one += 1
        if -1 in changed_idxs:
            structural_changes += 1

        idx_match = (changed_idxs == [claimed_idx]) if one_sentence_changed else False
        if idx_match:
            idx_matches += 1

        status = "OK" if (one_sentence_changed and idx_match) else "FAIL"
        print(f"\n[{i+1:02d}/{len(sample)}] {status}")
        print(f"  Prompt (truncated): {r['prompt'][:120]!r}")
        orig_sents = segment(orig)
        if 0 <= claimed_idx < len(orig_sents):
            print(f"  Original sentence [{claimed_idx}]: {orig_sents[claimed_idx]!r}")
        pert_sents = segment(pert)
        if 0 <= claimed_idx < len(pert_sents):
            print(f"  Perturbed sentence[{claimed_idx}]: {pert_sents[claimed_idx]!r}")
        if not one_sentence_changed:
            print(f"  WARNING: {n_changed} sentences changed (expected 1), changed_idxs={changed_idxs}")
        if one_sentence_changed and not idx_match:
            print(f"  WARNING: claimed idx={claimed_idx} but actual changed idx={changed_idxs}")

    print(f"\n--- {path.stem} summary ---")
    print(f"  Exactly one sentence changed : {exactly_one}/{len(sample)}")
    print(f"  Claimed idx matches actual   : {idx_matches}/{len(sample)}")
    print(f"  Structural changes (add/rm)  : {structural_changes}/{len(sample)}")

    return {
        "constitution": path.stem,
        "n_audited": len(sample),
        "exactly_one_changed": exactly_one,
        "idx_matches": idx_matches,
        "structural_changes": structural_changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality audit for perturbation datasets (step 43).")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--n", type=int, default=20, help="Examples per constitution")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = []
    for constitution in CONSTITUTIONS:
        path = args.input_dir / f"{constitution}.jsonl"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue
        result = audit_constitution(path, args.n, args.seed)
        results.append(result)

    print(f"\n{'='*70}")
    print("AUDIT SUMMARY")
    print(f"{'='*70}")
    print(f"{'Constitution':<30} {'1-sent-changed':>14} {'idx-match':>10} {'structural':>11}")
    for r in results:
        n = r["n_audited"]
        print(
            f"  {r['constitution']:<28} "
            f"{r['exactly_one_changed']:>5}/{n:<8} "
            f"{r['idx_matches']:>5}/{n:<4} "
            f"{r['structural_changes']:>5}/{n}"
        )


if __name__ == "__main__":
    main()
