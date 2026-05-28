"""Qwen Vision LLM implementation (OpenAI-compatible)."""

from __future__ import annotations

from typing import Any, Optional

from src.libs.llm.openai_vision_llm import OpenAIVisionLLM


class QwenVisionLLMError(RuntimeError):
    """Raised when Qwen vision API call fails."""


class QwenVisionLLM(OpenAIVisionLLM):
    """Qwen Vision provider using DashScope OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        settings: Any,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_image_size: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        vision_settings = getattr(settings, "vision_llm", None)
        selected_base_url = (
            base_url
            or (getattr(vision_settings, "base_url", None) if vision_settings else None)
            or self.DEFAULT_BASE_URL
        )
        super().__init__(
            settings=settings,
            api_key=api_key,
            base_url=selected_base_url,
            max_image_size=max_image_size,
            **kwargs,
        )
