"""Runtime guard to enforce the project virtual environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_expected_venv(
    entrypoint: str,
    expected_dir: str = ".venv311",
    allow_bypass_env: str = "MODULAR_RAG_SKIP_ENV_GUARD",
) -> None:
    """Exit fast if current interpreter is not the expected venv.

    Args:
        entrypoint: Script/module path displayed in guidance.
        expected_dir: Expected virtualenv directory name in repo root.
        allow_bypass_env: Env var name to bypass this guard when set to "1".
    """
    if os.getenv(allow_bypass_env) == "1":
        return

    exe_path = Path(sys.executable).resolve()
    expected_segment = str(Path(expected_dir) / "Scripts").lower()
    exe_lower = str(exe_path).lower()

    if expected_segment in exe_lower:
        return

    expected_python = Path(expected_dir) / "Scripts" / "python.exe"
    message = (
        f"[ENV ERROR] Expected interpreter from '{expected_dir}', got:\n"
        f"  {exe_path}\n\n"
        f"Please run with:\n"
        f"  {expected_python} {entrypoint}\n\n"
        f"Temporary bypass (not recommended):\n"
        f"  set {allow_bypass_env}=1"
    )
    raise SystemExit(message)

