from __future__ import annotations

import os

from huggingface_hub import get_token


def resolve_hf_token(env_var: str = "HF_TOKEN") -> str | None:
    """Resolve a Hugging Face token from env first, then local HF login."""
    token = os.environ.get(env_var)
    if token:
        return token
    return get_token()


def hf_storage_options(token: str | None) -> dict[str, str]:
    return {"token": token} if token else {}
