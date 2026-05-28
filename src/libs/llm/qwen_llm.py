"""Qwen LLM implementation (OpenAI-compatible)."""

from __future__ import annotations

from typing import Any, Optional

from src.libs.llm.openai_llm import OpenAILLM


class QwenLLMError(RuntimeError):
    """Raised when Qwen API call fails."""


class QwenLLM(OpenAILLM):
    """Qwen LLM provider using DashScope OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        selected_base_url = base_url or getattr(settings.llm, "base_url", None) or self.DEFAULT_BASE_URL
        super().__init__(
            settings=settings,
            api_key=api_key,
            base_url=selected_base_url,
            **kwargs,
        )
