"""Load and format prompts for grounded answer generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.core.settings import resolve_path
from src.libs.llm.base_llm import Message

DEFAULT_ANSWER_PROMPT_PATH = "config/prompts/answer_generation.txt"
_SYSTEM_MARKER = "===SYSTEM==="
_USER_MARKER = "===USER==="

_cached_templates: Optional[Tuple[str, str]] = None
_cached_path: Optional[str] = None


def _parse_prompt_file(content: str) -> Tuple[str, str]:
    """Split prompt file into (system, user_template) sections."""
    if _SYSTEM_MARKER not in content or _USER_MARKER not in content:
        raise ValueError(
            f"Answer prompt must contain {_SYSTEM_MARKER} and {_USER_MARKER} sections"
        )
    _, after_system = content.split(_SYSTEM_MARKER, 1)
    system_part, user_part = after_system.split(_USER_MARKER, 1)
    system = system_part.strip()
    user_template = user_part.strip()
    if not system or not user_template:
        raise ValueError("Answer prompt system and user sections must be non-empty")
    return system, user_template


def load_answer_prompt_templates(prompt_path: Optional[str] = None) -> Tuple[str, str]:
    """Load system and user prompt templates from file (cached per path)."""
    global _cached_templates, _cached_path
    path = str(
        resolve_path(prompt_path or DEFAULT_ANSWER_PROMPT_PATH)
    )
    if _cached_templates is not None and _cached_path == path:
        return _cached_templates

    prompt_file = Path(path)
    if not prompt_file.exists():
        raise FileNotFoundError(f"Answer generation prompt file not found: {path}")

    system, user_template = _parse_prompt_file(
        prompt_file.read_text(encoding="utf-8")
    )
    _cached_templates = (system, user_template)
    _cached_path = path
    return system, user_template


def resolve_answer_prompt_path(settings: Any = None) -> Optional[str]:
    """Return configured prompt path from settings, if any."""
    if settings is None:
        return None
    generation = getattr(settings, "generation", None)
    if generation is None:
        return None
    return getattr(generation, "prompt_path", None)


def build_answer_messages(
    query: str,
    context: str,
    *,
    prompt_path: Optional[str] = None,
    settings: Any = None,
) -> List[Message]:
    """Build LLM messages for grounded answer generation."""
    path = prompt_path or resolve_answer_prompt_path(settings)
    system, user_template = load_answer_prompt_templates(path)
    user_content = user_template.format(query=query, context=context)
    return [
        Message(role="system", content=system),
        Message(role="user", content=user_content),
    ]


def clear_answer_prompt_cache() -> None:
    """Clear cached templates (for tests)."""
    global _cached_templates, _cached_path
    _cached_templates = None
    _cached_path = None
