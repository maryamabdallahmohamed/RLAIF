import os
import time

import httpx


def chat(
    ollama_url: str,
    model: str,
    system_prompt: str,
    user_message: str,
    max_retries: int = 4,
    backoff_base: float = 2.0,
) -> str:
    """Call Ollama /api/chat against local or cloud Ollama, with retries.

    For Ollama Cloud, set OLLAMA_API_KEY in the environment and pass
    ollama_url="https://ollama.com". For local Ollama, leave the env var unset
    and use http://localhost:11434.

    Retries on httpx.TransportError (SSL/connection drops) and 5xx responses
    with exponential backoff. 4xx errors raise immediately.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }
    headers = {}
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = f"{ollama_url.rstrip('/')}/api/chat"
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = httpx.post(url, json=payload, headers=headers, timeout=180.0)
            # 429 Too Many Requests: honour Retry-After if present, else exponential.
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                try:
                    wait = float(ra) if ra else backoff_base ** (attempt + 1)
                except (TypeError, ValueError):
                    wait = backoff_base ** (attempt + 1)
                last_exc = httpx.HTTPStatusError(
                    "429 rate-limited", request=resp.request, response=resp
                )
                time.sleep(min(wait, 60.0))
                continue
            if 500 <= resp.status_code < 600:
                last_exc = httpx.HTTPStatusError(
                    f"server {resp.status_code}", request=resp.request, response=resp
                )
                time.sleep(backoff_base ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except httpx.TransportError as e:
            last_exc = e
            time.sleep(backoff_base ** attempt)
            continue
    assert last_exc is not None
    raise last_exc
