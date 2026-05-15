import httpx


def chat(
    ollama_url: str,
    model: str,
    system_prompt: str,
    user_message: str,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }
    resp = httpx.post(f"{ollama_url}/api/chat", json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.json()["message"]["content"]
