"""Environment/config access. Loads .env once at import."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_key(name: str) -> str:
    """Return a required env var or raise a clear error."""
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var {name!r}; set it in .env (see .env.example).")
    return val


def get_opt(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)
