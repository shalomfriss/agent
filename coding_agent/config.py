"""Command-line configuration and defaults."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


DEFAULT_BASE_URL = "http://127.0.0.1:8088/v1"
DEFAULT_MODEL = "models/gemma-4-e2b-it-4bit"
DEFAULT_OPENAI_MODEL = "gpt-6-luna"
DEFAULT_SESSION_DB = ".deep-agent/sessions.sqlite3"


@dataclass(frozen=True)
class AgentConfig:
    """Resolved settings for one agent run.

    Use :func:`parse_cli_arguments` to build this configuration from CLI
    options. OpenAI is selected when no provider is specified.

    Examples:
        >>> parse_cli_arguments([]).provider
        'openai'
        >>> config = parse_cli_arguments(
        ...     ["--provider", "mlx", "--model", "local-model", "Fix", "tests"]
        ... )
        >>> (config.provider, config.model_name, config.prompt)
        ('mlx', 'local-model', 'Fix tests')
    """

    # Directory whose files and shell commands the agent uses.
    project_root: Path

    # Model service selected by --provider: openai, anthropic, or mlx.
    provider: str

    # MLX or OpenAI-compatible endpoint; None uses the provider's own endpoint.
    base_url: str | None

    # Model ID sent to the selected provider.
    model_name: str

    # Sampling temperature; None leaves the provider's default in place.
    temperature: float | None

    # Whether to check chat and tool calling before starting a session.
    run_diagnostics: bool

    # Session ID to resume or create; None uses the last active session.
    requested_session: str | None

    # Whether to create a fresh session instead of resuming one.
    start_new_session: bool

    # Resolved path to the SQLite database that stores sessions.
    session_database: Path

    # One-shot user request; None starts the interactive shell.
    prompt: str | None


def parse_cli_arguments(arguments: Sequence[str] | None = None) -> AgentConfig:
    """Parse command-line arguments into a resolved, typed configuration."""

    parser = _build_argument_parser()
    parsed = parser.parse_args(arguments)
    if parsed.provider == "anthropic" and not parsed.model:
        parser.error("--model is required with --provider anthropic")
    if parsed.provider == "anthropic" and parsed.base_url:
        parser.error("--base-url is not supported with --provider anthropic")
    project_root = Path(parsed.project).expanduser().resolve()
    session_database = _resolve_session_database(
        value=parsed.session_db,
        project_root=project_root,
    )

    return AgentConfig(
        project_root=project_root,
        provider=parsed.provider,
        base_url=(
            parsed.base_url
            if parsed.base_url is not None
            else (DEFAULT_BASE_URL if parsed.provider == "mlx" else None)
        ),
        model_name=parsed.model or (
            DEFAULT_MODEL if parsed.provider == "mlx" else DEFAULT_OPENAI_MODEL
        ),
        temperature=(
            parsed.temperature
            if parsed.temperature is not None
            else (0.0 if parsed.provider == "mlx" else None)
        ),
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
        description="Run a coding agent with a local MLX, OpenAI, or Anthropic model."
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
        "--provider",
        choices=("mlx", "openai", "anthropic"),
        default="openai",
        help="Model provider. Default: openai",
    )
    parser.add_argument(
        "--base-url",
        help=f"MLX or OpenAI-compatible API URL. MLX default: {DEFAULT_BASE_URL}",
    )
    parser.add_argument(
        "--model",
        help=(
            f"Model ID. OpenAI default: {DEFAULT_OPENAI_MODEL}; "
            f"MLX default: {DEFAULT_MODEL}; required for Anthropic."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Model temperature. MLX default: 0; cloud provider default: provider setting.",
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
