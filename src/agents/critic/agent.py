import json
import re
from pathlib import Path
from typing import Callable

import yaml

from . import cache as _cache


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class CriticParseError(Exception):
    pass


class CriticAgent:
    def __init__(
        self,
        ollama_url: str,
        model: str,
        constitutions_path: Path,
        cache_dir: Path,
        _client_fn: Callable[[str, str, str, str], str] | None = None,
    ):
        self._cache_dir = cache_dir
        self._constitutions: dict = yaml.safe_load(constitutions_path.read_text())

        if _client_fn is None:
            from . import client as _client
            self._chat = lambda sys_p, usr: _client.chat(ollama_url, model, sys_p, usr)
        else:
            self._chat = lambda sys_p, usr: _client_fn(ollama_url, model, sys_p, usr)

    def perturb(
        self,
        prompt: str,
        response: str,
        constitution_name: str,
    ) -> tuple[str, int]:
        cached = _cache.read(self._cache_dir, constitution_name, prompt, response)
        if cached is not None:
            return cached

        if constitution_name not in self._constitutions:
            raise CriticParseError(
                f"Unknown constitution: {constitution_name!r}. "
                f"Available: {list(self._constitutions)}"
            )
        system_prompt = self._constitutions[constitution_name]["system_prompt"]
        user_message = f"Prompt: {prompt}\n\nResponse: {response}"
        raw = self._chat(system_prompt, user_message)

        try:
            json_str = raw.strip()
            m = _FENCE_RE.fullmatch(json_str)
            if m:
                json_str = m.group(1)
            data = json.loads(json_str)
            perturbed_response = str(data["perturbed_response"])
            sentence_idx = int(data["sentence_idx"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise CriticParseError(f"Failed to parse model output: {raw!r}") from exc

        _cache.write(
            self._cache_dir, constitution_name, prompt, response,
            perturbed_response, sentence_idx,
        )
        return perturbed_response, sentence_idx
