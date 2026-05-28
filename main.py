"""
MathRAG MCP Server - Main Entry Point

Loads configuration, configures logging, and starts the MCP stdio server.
"""

import sys
from pathlib import Path

from src.core.env_guard import ensure_expected_venv
from src.core.settings import SettingsError, load_settings
from src.observability.logger import configure_logging


def main() -> int:
    """CLI entry point (``mcp-server`` / ``python main.py``)."""
    ensure_expected_venv("main.py")

    settings_path = Path("config/settings.yaml")
    try:
        settings = load_settings(settings_path)
    except SettingsError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    configure_logging(settings.observability.log_level)

    from src.mcp_server.server import run_stdio_server

    return run_stdio_server()


if __name__ == "__main__":
    sys.exit(main())
