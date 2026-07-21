"""Command-line configuration and defaults."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:8088/v1"
DEFAULT_MODEL = "models/gemma-4-e2b-it-4bit"
DEFAULT_SESSION_DB = ".deep-agent/sessions.sqlite3"


@dataclass(frozen=True)
class AgentConfig:
    """All user-selectable settings needed to start the agent."""

    project_root: Path
    base_url: str
    model_name: str
    temperature: float
    run_diagnostics: bool
    requested_session: str | None
    start_new_session: bool
    session_database: Path
    prompt: str | None


def parse_cli_arguments(arguments: Sequence[str] | None = None) -> AgentConfig:
    """Parse command-line arguments into a resolved, typed configuration."""

    parser = _build_argument_parser()
    parsed = parser.parse_args(arguments)
    project_root = Path(parsed.project).expanduser().resolve()
    session_database = _resolve_session_database(
        value=parsed.session_db,
        project_root=project_root,
    )

    return AgentConfig(
        project_root=project_root,
        base_url=parsed.base_url,
        model_name=parsed.model,
        temperature=parsed.temperature,
        run_diagnostics=not parsed.skip_tests,
        requested_session=parsed.session,
        start_new_session=parsed.new_session,
        session_database=session_database,
        prompt=" ".join(parsed.prompt) if parsed.prompt else None,
    )


def _resolve_session_database(value: str, project_root: Path) -> Path:
    database = Path(value).expanduser()
    return database if database.is_absolute() else project_root / database


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local coding agent against an MLX model endpoint."
    )
    parser.add_argument(
        "-C",
        "--cd",
        "--project",
        dest="project",
        default=".",
        help=(
            "Run with this directory as the project root, like codex -C. "
            "Default: current directory."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI-compatible API URL. Default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model ID exposed by the server. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature. Default: 0",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip basic-chat and structured-tool-call diagnostics.",
    )

    session_options = parser.add_mutually_exclusive_group()
    session_options.add_argument(
        "--session",
        help="Resume or create a session with this ID.",
    )
    session_options.add_argument(
        "--new-session",
        action="store_true",
        help="Start a new session instead of resuming the last active one.",
    )

    parser.add_argument(
        "--session-db",
        default=DEFAULT_SESSION_DB,
        help=(
            "SQLite checkpoint path, relative to the project by default. "
            f"Default: {DEFAULT_SESSION_DB}"
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional one-shot prompt. Without this, an interactive CLI starts.",
    )
    return parser
