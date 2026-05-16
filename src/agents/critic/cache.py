import hashlib
import json
from pathlib import Path


def _cache_key(constitution_name: str, prompt: str, response: str, perturbation_idx: int = 0) -> str:
    payload = json.dumps([constitution_name, prompt, response, perturbation_idx], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def read(
    cache_dir: Path,
    constitution_name: str,
    prompt: str,
    response: str,
    perturbation_idx: int = 0,
) -> tuple[str, int] | None:
    key = _cache_key(constitution_name, prompt, response, perturbation_idx)
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data["perturbed_response"], data["sentence_idx"]
    except (json.JSONDecodeError, KeyError):
        return None


def write(
    cache_dir: Path,
    constitution_name: str,
    prompt: str,
    response: str,
    perturbed_response: str,
    sentence_idx: int,
    perturbation_idx: int = 0,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(constitution_name, prompt, response, perturbation_idx)
    path = cache_dir / f"{key}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"perturbed_response": perturbed_response, "sentence_idx": sentence_idx})
    )
    tmp.rename(path)
