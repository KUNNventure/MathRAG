"""Qwen Embedding implementation (OpenAI-compatible)."""

from __future__ import annotations

from typing import Any, Optional

from src.libs.embedding.openai_embedding import OpenAIEmbedding


class QwenEmbeddingError(RuntimeError):
    """Raised when Qwen embedding API call fails."""


class QwenEmbedding(OpenAIEmbedding):
    """Qwen embedding provider using DashScope OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        selected_base_url = (
            base_url
            or getattr(settings.embedding, "base_url", None)
            or self.DEFAULT_BASE_URL
        )
        super().__init__(
            settings=settings,
            api_key=api_key,
            base_url=selected_base_url,
            **kwargs,
        )
