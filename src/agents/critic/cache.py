import hashlib
import json
from pathlib import Path


def _cache_key(constitution_name: str, prompt: str, response: str) -> str:
    payload = constitution_name + prompt + response
    return hashlib.sha256(payload.encode()).hexdigest()


def read(
    cache_dir: Path,
    constitution_name: str,
    prompt: str,
    response: str,
) -> tuple[str, int] | None:
    key = _cache_key(constitution_name, prompt, response)
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data["perturbed_response"], data["sentence_idx"]


def write(
    cache_dir: Path,
    constitution_name: str,
    prompt: str,
    response: str,
    perturbed_response: str,
    sentence_idx: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(constitution_name, prompt, response)
    path = cache_dir / f"{key}.json"
    path.write_text(
        json.dumps({"perturbed_response": perturbed_response, "sentence_idx": sentence_idx})
    )
