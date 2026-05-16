"""
Schema-level smoke tests for dpo.preprocessing.

These tests do NOT hit the network and do NOT load a tokenizer. They cover
the pure-Python `_extract_pair` / `_completion_score` logic so we can run
them on the CPU before burning a Colab session.
"""
from __future__ import annotations

import pytest

from dpo.preprocessing import _completion_score, _extract_pair


def _completion(score: float | None, response: str = "ok") -> dict:
    return {"overall_score": score, "response": response}


def _annotated(rating: int, response: str = "ok") -> dict:
    """Completion with no overall_score but full annotation dims."""
    return {
        "response": response,
        "annotations": {
            dim: {"Rating": str(rating)}
            for dim in ("instruction_following", "helpfulness", "honesty", "truthfulness")
        },
    }


def test_completion_score_prefers_overall_score():
    assert _completion_score(_completion(4.5)) == pytest.approx(4.5)


def test_completion_score_falls_back_to_annotations():
    assert _completion_score(_annotated(3)) == pytest.approx(3.0)


def test_completion_score_returns_none_when_unscoreable():
    assert _completion_score({"response": "no score anywhere"}) is None


def test_extract_pair_basic():
    sample = {
        "instruction": "Explain photosynthesis.",
        "completions": [
            _completion(2.0, "bad answer"),
            _completion(4.5, "great answer"),
            _completion(3.0, "middling answer"),
        ],
    }
    pair = _extract_pair(sample, min_score_gap=0.5)
    assert pair is not None
    assert pair["prompt"] == "Explain photosynthesis."
    assert pair["chosen"] == "great answer"
    assert pair["rejected"] == "bad answer"
    assert pair["score_gap"] == pytest.approx(2.5)


def test_extract_pair_drops_when_gap_below_threshold():
    sample = {
        "instruction": "Q?",
        "completions": [
            _completion(3.0, "a"),
            _completion(3.2, "b"),
        ],
    }
    assert _extract_pair(sample, min_score_gap=0.5) is None


def test_extract_pair_drops_identical_responses():
    sample = {
        "instruction": "Q?",
        "completions": [
            _completion(1.0, "same text"),
            _completion(5.0, "same text"),
        ],
    }
    assert _extract_pair(sample, min_score_gap=0.5) is None


def test_extract_pair_drops_missing_prompt():
    sample = {
        "instruction": "",
        "completions": [
            _completion(1.0, "x"),
            _completion(5.0, "y"),
        ],
    }
    assert _extract_pair(sample, min_score_gap=0.5) is None


def test_extract_pair_requires_two_scoreable_completions():
    sample = {
        "instruction": "Q?",
        "completions": [_completion(5.0, "only one")],
    }
    assert _extract_pair(sample, min_score_gap=0.5) is None


def test_extract_pair_handles_annotation_fallback():
    sample = {
        "instruction": "Compare A and B.",
        "completions": [
            _annotated(1, "weak"),
            _annotated(5, "strong"),
        ],
    }
    pair = _extract_pair(sample, min_score_gap=1.0)
    assert pair is not None
    assert pair["chosen"] == "strong"
    assert pair["rejected"] == "weak"
