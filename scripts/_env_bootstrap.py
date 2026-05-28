"""Load ``<repo>/.env`` into os.environ for CLI scripts (does not modify src/)."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_repo_dotenv_loaded(repo_root: Path | None = None) -> bool:
    """Load ``.env`` with setdefault (existing env vars win). Returns True if file existed."""
    root = repo_root if repo_root is not None else _REPO_ROOT
    dotenv_path = root / ".env"
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
