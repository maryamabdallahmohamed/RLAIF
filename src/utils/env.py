"""Minimal .env loader (no python-dotenv dependency)."""
import os
from pathlib import Path


def load_env(path: str | Path = ".env") -> None:
    """Load KEY=VALUE pairs from .env into os.environ if not already set."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
