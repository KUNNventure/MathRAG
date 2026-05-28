#!/usr/bin/env python
"""Smoke test: repo .env loads DASHSCOPE_API_KEY before evaluate/rescore."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._env_bootstrap import ensure_repo_dotenv_loaded  # noqa: E402

DOTENV = PROJECT_ROOT / ".env"


def main() -> int:
    had_file = ensure_repo_dotenv_loaded()
    key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY")

    print(f"repo:     {PROJECT_ROOT}")
    print(f".env:     {DOTENV} ({'found' if had_file else 'missing'})")

    if not had_file:
        print("\nFAIL: create .env from .env.example and set DASHSCOPE_API_KEY")
        return 1
    if not key or not key.strip():
        print("\nFAIL: .env exists but DASHSCOPE_API_KEY / OPENAI_API_KEY is empty")
        return 1

    prefix = key.strip()[:7]
    print(f"env key:  set (len={len(key.strip())}, prefix={prefix}...)")

    from src.core.api_keys import resolve_api_key  # noqa: E402

    resolved = resolve_api_key(provider="qwen")
    if not resolved:
        print("\nFAIL: resolve_api_key(qwen) returned nothing after bootstrap")
        return 1
    print("resolve:  OK (qwen)")

    try:
        from src.core.settings import load_settings  # noqa: E402
        from src.libs.embedding.embedding_factory import EmbeddingFactory  # noqa: E402

        settings = load_settings()
        EmbeddingFactory.create(settings)
        print("embedding: OK (provider instantiated)")
    except Exception as exc:
        print(f"\nFAIL: embedding client: {exc}")
        return 1

    print("\nOK: .env bootstrap works for evaluate.py stack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
