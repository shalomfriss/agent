"""Application lifecycle and interactive terminal commands."""

from __future__ import annotations

import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from typing import Callable, Iterable

from langgraph.checkpoint.sqlite import SqliteSaver
from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import (
    CompleteEvent,
    Completer,
    Completion,
    WordCompleter,
)
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style
from rich.markup import escape

from coding_agent.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    AgentConfig,
    parse_cli_arguments,
)
from coding_agent.runtime import (
    DiagnosticResult,
    create_coding_agent,
    create_model,
    run_model_diagnostics,
)
from coding_agent.sessions import SessionManager
from coding_agent.streaming import AgentRunner
from coding_agent.terminal import AgentIdentity, TerminalDisplay
from coding_agent.workspace import Workspace


EXIT_SUCCESS = 0
EXIT_CONFIGURATION_ERROR = 1
MODEL_PROVIDERS = ("openai", "anthropic", "mlx")
ModelSwitcher = Callable[[str, str], None]


@dataclass(frozen=True)
class SlashCommand:
    """One interactive command and the text shown in its completion menu."""

    completion: str
    display: str
    description: str


SLASH_COMMANDS = (
    SlashCommand("/new", "/new", "Start a new conversation"),
    SlashCommand("/sessions", "/sessions", "List saved conversations"),
    SlashCommand("/switch ", "/switch SESSION_ID", "Change conversations"),
    SlashCommand("/model ", "/model [PROVIDER:]MODEL", "Choose a model"),
    SlashCommand("/status", "/status", "Show model and session details"),
    SlashCommand("/clear", "/clear", "Clear the terminal"),
    SlashCommand("/help", "/help", "Show commands and shortcuts"),
    SlashCommand("/quit", "/quit", "Exit the agent"),
    SlashCommand("/exit", "/exit", "Exit the agent"),
)


PROMPT_STYLE = Style.from_dict(
    {
        "prompt": "bold #a78bfa",
        "toolbar.label": "bg:#111827 #94a3b8",
        "toolbar.value": "bg:#111827 bold #e2e8f0",
        "toolbar.dim": "bg:#111827 #64748b",
        "completion-menu.completion": "bg:#111827 #e2e8f0",
        "completion-menu.completion.current": "bg:#4c1d95 #ffffff bold",
        "completion-menu.meta.completion": "bg:#111827 #94a3b8",
        "completion-menu.meta.completion.current": "bg:#4c1d95 #ddd6fe",
    }
)


class SlashCommandCompleter(Completer):
    """Offer slash commands without completing ordinary prompt text."""

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        del complete_event
        prefix = document.text_before_cursor
        if not prefix.startswith("/") or " " in prefix:
            return

        normalized = prefix.lower()
        for command in SLASH_COMMANDS:
            if command.completion.startswith(normalized):
                yield Completion(
                    command.completion,
                    start_position=-len(prefix),
                    display=command.display,
                    display_meta=command.description,
                )


def create_slash_command_key_bindings() -> KeyBindings:
    """Add command cancellation and multiline editing shortcuts."""

    bindings = KeyBindings()
    slash_command_active = Condition(
        lambda: get_app().current_buffer.text.startswith("/")
    )

    @bindings.add(Keys.Escape, filter=slash_command_active, eager=True)
    def clear_slash_command(event: KeyPressEvent) -> None:
        event.current_buffer.reset()

    @bindings.add(Keys.Escape, Keys.Enter)
    def insert_newline(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    return bindings


def parse_model_selection(
    selection: str,
    current_provider: str,
) -> tuple[str, str]:
    """Resolve MODEL or PROVIDER:MODEL input into validated components."""

    value = selection.strip()
    if not value:
        raise ValueError("Model ID cannot be empty.")

    provider = current_provider
    model = value
    prefix, separator, remainder = value.partition(":")
    if separator and prefix.lower() in MODEL_PROVIDERS:
        provider = prefix.lower()
        model = remainder.strip()
    if provider not in MODEL_PROVIDERS:
        raise ValueError(f"Unknown model provider: {provider}")
    if not model:
        raise ValueError("Model ID cannot be empty.")
    return provider, model


class CodingAgentApplication:
    """Prepare dependencies and run either a one-shot or interactive agent."""

    def __init__(self, config: AgentConfig):
        self._config = config

    def run(self) -> int:
        try:
            workspace = Workspace.prepare(self._config)
        except (OSError, ValueError) as error:
            print(error, file=sys.stderr)
            return EXIT_CONFIGURATION_ERROR

        try:
            model = create_model(
                provider=self._config.provider,
                base_url=self._config.base_url,
                model_name=self._config.model_name,
                temperature=self._config.temperature,
            )
        except ValueError as error:
            print(error, file=sys.stderr)
            return EXIT_CONFIGURATION_ERROR

        if self._config.run_diagnostics:
            diagnostic_result = run_model_diagnostics(model)
            if diagnostic_result is not DiagnosticResult.PASSED:
                return int(diagnostic_result)

        connection = sqlite3.connect(
            workspace.session_database,
            check_same_thread=False,
        )
        with closing(connection):
            sessions = SessionManager(connection, workspace.project_root)
            session_id, resumed = sessions.activate(
                self._config.requested_session,
                force_new=self._config.start_new_session,
            )
            agent = create_coding_agent(
                model=model,
                project_root=workspace.project_root,
                checkpointer=SqliteSaver(connection),
            )
            display = TerminalDisplay(
                AgentIdentity(
                    project_root=workspace.project_root,
                    provider=self._config.provider,
                    model=self._config.model_name,
                    session_id=session_id,
                    resumed=resumed,
                )
            )
            runner = AgentRunner(agent, console=display.console)

            def switch_model(provider: str, model_name: str) -> None:
                base_url = self._provider_base_url(provider)
                temperature = self._provider_temperature(provider)
                replacement_model = create_model(
                    provider=provider,
                    base_url=base_url,
                    model_name=model_name,
                    temperature=temperature,
                )
                replacement_agent = create_coding_agent(
                    model=replacement_model,
                    project_root=workspace.project_root,
                    checkpointer=SqliteSaver(connection),
                )
                runner.replace_agent(replacement_agent)

            display.welcome()
            if self._config.prompt is not None:
                return self._run_once(
                    runner,
                    sessions,
                    session_id,
                    self._config.prompt,
                )

            shell = InteractiveShell(
                runner,
                sessions,
                display,
                switch_model,
            )
            return shell.run()

    def _run_once(
        self,
        runner: AgentRunner,
        sessions: SessionManager,
        session_id: str,
        prompt: str,
    ) -> int:
        try:
            runner.run(prompt, session_id)
        finally:
            sessions.touch(session_id)
        return EXIT_SUCCESS

    def _provider_base_url(self, provider: str) -> str | None:
        if provider == self._config.provider:
            return self._config.base_url
        return DEFAULT_BASE_URL if provider == "mlx" else None

    def _provider_temperature(self, provider: str) -> float | None:
        if provider == self._config.provider:
            return self._config.temperature
        return 0.0 if provider == "mlx" else None


class InteractiveShell:
    """Read prompts and implement the agent's slash commands."""

    EXIT_COMMANDS = {"/quit", "/exit", "quit", "exit"}

    def __init__(
        self,
        runner: AgentRunner,
        sessions: SessionManager,
        display: TerminalDisplay,
        model_switcher: ModelSwitcher,
    ):
        self._runner = runner
        self._sessions = sessions
        self._display = display
        self._model_switcher = model_switcher
        self._session_id = display.identity.session_id
        self._prompt_session: PromptSession[str] | None = None
        if sys.stdin.isatty() and sys.stdout.isatty():
            self._prompt_session = PromptSession(
                completer=SlashCommandCompleter(),
                complete_while_typing=True,
                complete_style=CompleteStyle.COLUMN,
                key_bindings=create_slash_command_key_bindings(),
                bottom_toolbar=self._display.toolbar,
                style=PROMPT_STYLE,
            )

    def run(self) -> int:
        while True:
            prompt = self._read_prompt()
            if prompt is None:
                return EXIT_SUCCESS
            if not prompt:
                continue
            if prompt.lower() in self.EXIT_COMMANDS:
                self._display.notice("Goodbye.")
                return EXIT_SUCCESS
            if self._handle_command(prompt):
                continue

            try:
                self._runner.run(prompt, self._session_id)
            except KeyboardInterrupt:
                self._display.warning("Turn interrupted")
            except Exception:
                # A failed turn should not terminate the interactive shell.
                pass
            finally:
                self._sessions.touch(self._session_id)

    def _handle_command(self, prompt: str) -> bool:
        normalized = prompt.lower()
        if normalized == "/new":
            self._start_new_session()
            return True
        if normalized == "/sessions":
            self._list_sessions()
            return True
        if normalized == "/model":
            self._choose_model()
            return True
        if normalized.startswith("/model "):
            self._switch_model(prompt.split(maxsplit=1)[1])
            return True
        if normalized == "/status":
            self._display.status()
            return True
        if normalized == "/clear":
            self._display.clear()
            return True
        if normalized == "/help":
            self._display.help()
            return True
        if normalized.startswith("/switch "):
            self._switch_session(prompt.split(maxsplit=1)[1])
            return True
        return False

    def _start_new_session(self) -> None:
        self._session_id, _ = self._sessions.activate(force_new=True)
        self._display.set_session(self._session_id, resumed=False)

    def _switch_session(self, requested_id: str) -> None:
        self._session_id, existed = self._sessions.activate(requested_id)
        self._display.set_session(self._session_id, resumed=existed)

    def _list_sessions(self) -> None:
        self._display.console.print("\n[bold #a78bfa]Saved threads[/]")
        for session_id, updated_at in self._sessions.list_sessions():
            marker = "◆" if session_id == self._session_id else "·"
            self._display.console.print(
                f"  {marker} {escape(session_id)}  "
                f"[#64748b]{escape(updated_at)}[/]"
            )

    def _choose_model(self) -> None:
        """Prompt for provider and model while keeping the current thread."""

        current_provider = self._display.identity.provider
        try:
            provider = self._read_choice(
                "provider",
                default=current_provider,
                choices=MODEL_PROVIDERS,
            ).lower()
            if provider not in MODEL_PROVIDERS:
                raise ValueError(f"Unknown model provider: {provider}")
            model = self._read_choice(
                "model",
                default=self._default_model(provider),
            )
            self._switch_model(f"{provider}:{model}")
        except (EOFError, KeyboardInterrupt):
            self._display.warning("Model selection cancelled")
        except ValueError as error:
            self._display.warning(str(error))

    def _switch_model(self, selection: str) -> None:
        try:
            provider, model = parse_model_selection(
                selection,
                self._display.identity.provider,
            )
            self._model_switcher(provider, model)
        except Exception as error:
            self._display.warning(str(error))
            return
        self._display.set_model(provider, model)

    def _default_model(self, provider: str) -> str:
        if provider == self._display.identity.provider:
            return self._display.identity.model
        if provider == "openai":
            return DEFAULT_OPENAI_MODEL
        if provider == "mlx":
            return DEFAULT_MODEL
        return ""

    def _read_choice(
        self,
        label: str,
        *,
        default: str,
        choices: tuple[str, ...] = (),
    ) -> str:
        message = [("class:prompt", f"{label} ❯ ")]
        if self._prompt_session is not None:
            completer = None
            if choices:
                completer = WordCompleter(choices, ignore_case=True)
            return self._prompt_session.prompt(
                message,
                default=default,
                completer=completer,
            ).strip()
        value = input(f"{label} [{default}] ❯ ").strip()
        return value or default

    def _read_prompt(self) -> str | None:
        try:
            if self._prompt_session is not None:
                return self._prompt_session.prompt(
                    [("class:prompt", "\n❯ ")]
                ).strip()
            return input("\n❯ ").strip()
        except KeyboardInterrupt:
            self._display.warning("Input cleared")
            return ""
        except EOFError:
            self._display.notice("Goodbye.")
            return None


def main() -> int:
    """Parse configuration and run the coding-agent application."""

    return CodingAgentApplication(parse_cli_arguments()).run()
