"""Resolve API keys from settings, environment variables, or .env (never commit secrets)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

from src.core.settings import REPO_ROOT

# Provider → env var names (checked in order, after optional yaml/explicit key).
_PROVIDER_ENV_VARS: dict[str, tuple[str, ...]] = {
    "qwen": ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"),
    "openai": ("OPENAI_API_KEY", "DASHSCOPE_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "azure": ("AZURE_OPENAI_API_KEY",),
}


def _non_empty(value: Optional[str]) -> Optional[str]:
    if value is None or not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def load_dotenv_if_present(path: Optional[Path] = None) -> bool:
    """Load ``<repo>/.env`` into ``os.environ`` (setdefault; does not override existing)."""
    dotenv_path = path if path is not None else REPO_ROOT / ".env"
    if not dotenv_path.is_file():
        return False

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return True


def _env_candidates(provider: Optional[str]) -> Sequence[str]:
    names: list[str] = []
    if provider:
        names.extend(_PROVIDER_ENV_VARS.get(provider.lower(), ()))
    for fallback in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
        if fallback not in names:
            names.append(fallback)
    return names


def resolve_api_key(
    yaml_key: Optional[str] = None,
    *,
    provider: Optional[str] = None,
    explicit: Optional[str] = None,
) -> Optional[str]:
    """Return API key from explicit override, non-empty yaml value, or environment."""
    key = _non_empty(explicit) or _non_empty(yaml_key)
    if key:
        return key
    for env_name in _env_candidates(provider):
        found = _non_empty(os.environ.get(env_name))
        if found:
            return found
    return None


def require_api_key(
    yaml_key: Optional[str] = None,
    *,
    provider: Optional[str] = None,
    explicit: Optional[str] = None,
    service: str = "API",
) -> str:
    """Like :func:`resolve_api_key` but raises if no key is available."""
    key = resolve_api_key(yaml_key, provider=provider, explicit=explicit)
    if key:
        return key
    env_hint = ", ".join(_env_candidates(provider))
    raise ValueError(
        f"{service} key not provided. Set one of ({env_hint}), "
        f"or add DASHSCOPE_API_KEY / OPENAI_API_KEY to {REPO_ROOT / '.env'}, "
        "or pass api_key explicitly. Do not commit secrets in settings.yaml."
    )
