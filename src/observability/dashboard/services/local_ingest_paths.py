"""Discover ingestible files under a path (Dashboard + CLI-aligned extensions)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

DEFAULT_EXTENSIONS: tuple[str, ...] = (".pdf",)


def _norm_ext(e: str) -> str:
    e = e.strip().lower()
    return e if e.startswith(".") else f".{e}"


def discover_ingest_files(
    root: Path,
    extensions: Sequence[str] | None = None,
    *,
    recursive: bool = True,
) -> List[Path]:
    """Return sorted unique files under ``root`` matching extensions.

    - ``root`` may be a file or directory.
    - If ``recursive`` is False, only the immediate directory is scanned.
    """
    exts = tuple(_norm_ext(x) for x in (extensions or DEFAULT_EXTENSIONS))
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(str(root))
    if root.is_file():
        suf = root.suffix.lower()
        if suf in exts:
            return [root]
        raise ValueError(f"Unsupported file type: {suf}. Supported: {exts}")

    seen: set[Path] = set()
    out: List[Path] = []
    for ext in exts:
        pattern = f"*{ext}"
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        for p in iterator:
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    out.sort()
    return out
