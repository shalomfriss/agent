"""Load JSON defaults and resolve immutable, provider-aware agent settings.

Editable defaults live in ``defaults.json`` beside this module. This module
owns typed loading and validation only; filesystem initialization and runtime
construction are handled by :mod:`coding_agent.workspace` and
:mod:`coding_agent.runtime` respectively.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULTS_PATH = Path(__file__).with_name("defaults.json")


@dataclass(frozen=True)
class ProviderDefaults:
    """Default connection and sampling values for one model provider."""

    model: str | None
    base_url: str | None
    temperature: float | None


@dataclass(frozen=True)
class ModelOptions:
    """Options passed to every LangChain chat-model client."""

    max_tokens: int
    max_retries: int
    timeout: float


@dataclass(frozen=True)
class BackendOptions:
    """Execution limits for the project-local shell backend."""

    command_timeout: float


@dataclass(frozen=True)
class RuntimeDefaults:
    """Model-client and backend runtime defaults."""

    model: ModelOptions
    backend: BackendOptions


@dataclass(frozen=True)
class SessionDefaults:
    """Persistence defaults for conversation checkpoints."""

    database: str


@dataclass(frozen=True)
class TerminalDefaults:
    """Timing defaults for terminal input handling."""

    escape_timeout: float


@dataclass(frozen=True)
class ApplicationDefaults:
    """Typed representation of ``defaults.json``."""

    providers: dict[str, ProviderDefaults]
    runtime: RuntimeDefaults
    sessions: SessionDefaults
    terminal: TerminalDefaults


def load_defaults(path: Path = DEFAULTS_PATH) -> ApplicationDefaults:
    """Load and validate the sectioned JSON defaults file."""

    with path.open(encoding="utf-8") as defaults_file:
        data: dict[str, Any] = json.load(defaults_file)

    try:
        providers = {
            name: ProviderDefaults(
                model=values.get("model"),
                base_url=values.get("base_url"),
                temperature=values.get("temperature"),
            )
            for name, values in data["providers"].items()
        }
        model_values = data["runtime"]["model"]
        backend_values = data["runtime"]["backend"]
        defaults = ApplicationDefaults(
            providers=providers,
            runtime=RuntimeDefaults(
                model=ModelOptions(
                    max_tokens=int(model_values["max_tokens"]),
                    max_retries=int(model_values["max_retries"]),
                    timeout=float(model_values["timeout"]),
                ),
                backend=BackendOptions(
                    command_timeout=float(backend_values["command_timeout"]),
                ),
            ),
            sessions=SessionDefaults(database=str(data["sessions"]["database"])),
            terminal=TerminalDefaults(
                escape_timeout=float(data["terminal"]["escape_timeout"]),
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid defaults file {path}: {error}") from error

    required_providers = {"openai", "anthropic", "mlx"}
    missing = required_providers.difference(defaults.providers)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"Invalid defaults file {path}: missing providers: {names}")
    if defaults.runtime.model.max_tokens <= 0:
        raise ValueError("Model max_tokens must be positive")
    if defaults.runtime.model.max_retries < 0:
        raise ValueError("Model max_retries cannot be negative")
    if defaults.runtime.model.timeout <= 0:
        raise ValueError("Model timeout must be positive")
    if defaults.runtime.backend.command_timeout <= 0:
        raise ValueError("Backend command_timeout must be positive")
    if defaults.terminal.escape_timeout < 0:
        raise ValueError("Terminal escape_timeout cannot be negative")
    return defaults


DEFAULTS = load_defaults()


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

    # Full defaults used to construct model, backend, and terminal behavior.
    defaults: ApplicationDefaults

    # One-shot user request; None starts the interactive shell.
    prompt: str | None


def parse_cli_arguments(
    arguments: Sequence[str] | None = None,
    *,
    defaults: ApplicationDefaults = DEFAULTS,
) -> AgentConfig:
    """Parse arguments and resolve paths and provider-specific defaults.

    Args:
        arguments: Explicit arguments for tests or embedding. ``None`` reads
            from ``sys.argv`` through :mod:`argparse`.

    Returns:
        A validated configuration with an absolute project path and resolved
        session database path.
    """

    parser = _build_argument_parser(defaults)
    parsed = parser.parse_args(arguments)
    if parsed.provider == "anthropic" and not parsed.model:
        parser.error("--model is required with --provider anthropic")
    if parsed.provider == "anthropic" and parsed.base_url:
        parser.error("--base-url is not supported with --provider anthropic")
    project_root = Path(parsed.project).expanduser().resolve()
    provider_defaults = defaults.providers[parsed.provider]
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
            else provider_defaults.base_url
        ),
        model_name=parsed.model or provider_defaults.model or "",
        temperature=(
            parsed.temperature
            if parsed.temperature is not None
            else provider_defaults.temperature
        ),
        run_diagnostics=not parsed.skip_tests,
        requested_session=parsed.session,
        start_new_session=parsed.new_session,
        session_database=session_database,
        defaults=defaults,
        prompt=" ".join(parsed.prompt) if parsed.prompt else None,
    )


def _resolve_session_database(value: str, project_root: Path) -> Path:
    """Resolve a session database path relative to the selected project."""

    database = Path(value).expanduser()
    return database if database.is_absolute() else project_root / database


def _build_argument_parser(
    defaults: ApplicationDefaults,
) -> argparse.ArgumentParser:
    """Define the command-line interface without parsing process arguments."""

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
        help=(
            "MLX or OpenAI-compatible API URL. "
            f"MLX default: {defaults.providers['mlx'].base_url}"
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            f"Model ID. OpenAI default: {defaults.providers['openai'].model}; "
            f"MLX default: {defaults.providers['mlx'].model}; "
            "required for Anthropic."
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
        default=defaults.sessions.database,
        help=(
            "SQLite checkpoint path, relative to the project by default. "
            f"Default: {defaults.sessions.database}"
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional one-shot prompt. Without this, an interactive CLI starts.",
    )
    return parser
