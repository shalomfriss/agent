"""Wire configuration, persistence, runtime execution, and terminal workflows.

``CodingAgentApplication`` owns process startup and dependency construction.
``InteractiveShell`` owns the long-lived prompt loop and slash-command state.
Keeping those responsibilities separate allows one-shot execution to reuse the
same model, checkpoint, and streaming components without terminal workflows.
"""

from __future__ import annotations

import sqlite3
import sys
from contextlib import closing

from langgraph.checkpoint.sqlite import SqliteSaver
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style

from coding_agent.commands import (
    CommandContext,
    CommandOutcome,
    ExitRequested,
    ModelLister,
    ModelSwitcher,
    SlashCommandCompleter,
    WorkflowCancelled,
    create_default_registry,
)
from coding_agent.config import (
    DEFAULTS,
    AgentConfig,
    ApplicationDefaults,
    parse_cli_arguments,
)
from coding_agent.runtime import (
    DiagnosticResult,
    create_coding_agent,
    create_model,
    list_available_models,
    run_model_diagnostics,
)
from coding_agent.sessions import SessionManager
from coding_agent.streaming import AgentRunner
from coding_agent.terminal import AgentIdentity, TerminalDisplay
from coding_agent.workspace import Workspace


EXIT_SUCCESS = 0
EXIT_CONFIGURATION_ERROR = 1


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


def create_slash_command_key_bindings() -> KeyBindings:
    """Add immediate, universal input cancellation shortcuts."""

    bindings = KeyBindings()

    @bindings.add(Keys.Escape, eager=True)
    def clear_input(event: KeyPressEvent) -> None:
        """Abort the current prompt immediately and discard its buffer."""

        event.current_buffer.reset()
        event.app.exit(exception=WorkflowCancelled())

    @bindings.add(Keys.ControlJ)
    def insert_newline(event: KeyPressEvent) -> None:
        """Insert a literal newline without submitting the prompt."""

        event.current_buffer.insert_text("\n")

    return bindings


def create_choice_key_bindings(choices: tuple[str, ...]) -> KeyBindings:
    """Cycle through a choice list and let Escape cancel the workflow."""

    bindings = KeyBindings()

    def cycle(event: KeyPressEvent, offset: int) -> None:
        """Replace the buffer with the adjacent choice, wrapping at either end."""

        if not choices:
            return
        current = event.current_buffer.text
        try:
            index = choices.index(current)
        except ValueError:
            index = -1 if offset > 0 else 0
        event.current_buffer.text = choices[(index + offset) % len(choices)]
        event.current_buffer.cursor_position = len(event.current_buffer.text)

    @bindings.add(Keys.Down)
    def next_choice(event: KeyPressEvent) -> None:
        """Select the next value in the active choice workflow."""

        cycle(event, 1)

    @bindings.add(Keys.Up)
    def previous_choice(event: KeyPressEvent) -> None:
        """Select the previous value in the active choice workflow."""

        cycle(event, -1)

    @bindings.add(Keys.Escape, eager=True)
    def cancel_choice(event: KeyPressEvent) -> None:
        """Unwind the complete nested workflow rather than clearing one field."""

        event.app.exit(exception=WorkflowCancelled())

    return bindings


class CodingAgentApplication:
    """Prepare dependencies and run either a one-shot or interactive agent."""

    def __init__(self, config: AgentConfig):
        """Store the fully resolved settings used for one process invocation."""

        self._config = config

    def run(self) -> int:
        """Build dependencies and execute one-shot or interactive mode.

        The SQLite connection remains open for the full agent lifetime because
        both session metadata and LangGraph checkpoints share it. Interactive
        model changes replace the compiled agent but intentionally reuse that
        connection and the active thread ID.
        """

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
                model_options=self._config.defaults.runtime.model,
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
                backend_options=self._config.defaults.runtime.backend,
                system_prompt_path=(
                    self._config.defaults.prompts.system_prompt_path
                ),
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
                """Rebuild the graph around a new model while retaining state."""

                base_url = self._provider_base_url(provider)
                temperature = self._provider_temperature(provider)
                replacement_model = create_model(
                    provider=provider,
                    base_url=base_url,
                    model_name=model_name,
                    temperature=temperature,
                    model_options=self._config.defaults.runtime.model,
                )
                replacement_agent = create_coding_agent(
                    model=replacement_model,
                    project_root=workspace.project_root,
                    checkpointer=SqliteSaver(connection),
                    backend_options=self._config.defaults.runtime.backend,
                    system_prompt_path=(
                        self._config.defaults.prompts.system_prompt_path
                    ),
                )
                runner.replace_agent(replacement_agent)

            def models_for_provider(provider: str) -> tuple[str, ...]:
                """Discover models using the endpoint rules from initial setup."""

                return list_available_models(
                    provider,
                    self._provider_base_url(provider),
                )

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
                models_for_provider,
                self._config.defaults,
            )
            return shell.run()

    def _run_once(
        self,
        runner: AgentRunner,
        sessions: SessionManager,
        session_id: str,
        prompt: str,
    ) -> int:
        """Run one prompt and always update the thread's recency metadata."""

        try:
            runner.run(prompt, session_id)
        finally:
            sessions.touch(session_id)
        return EXIT_SUCCESS

    def _provider_base_url(self, provider: str) -> str | None:
        """Select the original endpoint or the default endpoint for MLX."""

        if provider == self._config.provider:
            return self._config.base_url
        return self._config.defaults.providers[provider].base_url

    def _provider_temperature(self, provider: str) -> float | None:
        """Preserve configured sampling or apply the deterministic MLX default."""

        if provider == self._config.provider:
            return self._config.temperature
        return self._config.defaults.providers[provider].temperature


class InteractiveShell:
    """Coordinate prompts, slash commands, cancellation, and active runtime.

    Escape raises :class:`WorkflowCancelled` from prompt-toolkit workflows;
    Ctrl-C follows the explicit exit-confirmation path. A switched model keeps
    the same ``SessionManager`` thread, so checkpointed conversation state is
    available to the replacement agent.
    """

    TEXT_EXIT_COMMANDS = {"quit", "exit"}

    def __init__(
        self,
        runner: AgentRunner,
        sessions: SessionManager,
        display: TerminalDisplay,
        model_switcher: ModelSwitcher,
        model_lister: ModelLister | None = None,
        defaults: ApplicationDefaults = DEFAULTS,
    ):
        """Initialize shell collaborators and terminal-only prompt features."""

        self._runner = runner
        self._sessions = sessions
        self._display = display
        self._commands = create_default_registry()
        self._prompt_session: PromptSession[str] | None = None
        if sys.stdin.isatty() and sys.stdout.isatty():
            self._prompt_session = PromptSession(
                completer=SlashCommandCompleter(self._commands),
                complete_while_typing=True,
                complete_style=CompleteStyle.COLUMN,
                key_bindings=create_slash_command_key_bindings(),
                bottom_toolbar=self._display.toolbar,
                style=PROMPT_STYLE,
            )
            self._prompt_session.app.ttimeoutlen = (
                defaults.terminal.escape_timeout
            )
        self._command_context = CommandContext(
            sessions=sessions,
            display=display,
            session_id=display.identity.session_id,
            model_switcher=model_switcher,
            model_lister=model_lister or (lambda _provider: ()),
            defaults=defaults,
            read_choice=self._read_choice,
            confirm_exit=self._confirm_exit,
            help_rows=self._commands.help_rows(),
        )

    def run(self) -> int:
        """Read and dispatch input until EOF or an explicit exit request."""

        while True:
            prompt = self._read_prompt()
            if prompt is None:
                return EXIT_SUCCESS
            if not prompt:
                continue
            if prompt.lower() in self.TEXT_EXIT_COMMANDS:
                self._display.notice("Goodbye.")
                return EXIT_SUCCESS
            try:
                outcome = self._commands.dispatch(
                    prompt,
                    self._command_context,
                )
                if outcome is CommandOutcome.EXIT:
                    self._display.notice("Goodbye.")
                    return EXIT_SUCCESS
                if outcome is CommandOutcome.CONTINUE:
                    continue
            except KeyboardInterrupt:
                if self._confirm_exit():
                    self._display.notice("Goodbye.")
                    return EXIT_SUCCESS
                self._display.warning("Exit cancelled")
                continue
            except ExitRequested:
                self._display.notice("Goodbye.")
                return EXIT_SUCCESS

            try:
                self._runner.run(prompt, self._command_context.session_id)
            except KeyboardInterrupt:
                if self._confirm_exit():
                    self._display.notice("Goodbye.")
                    return EXIT_SUCCESS
                self._display.warning("Exit cancelled")
            except Exception:
                # A failed turn should not terminate the interactive shell.
                pass
            finally:
                self._sessions.touch(self._command_context.session_id)

    def _read_choice(
        self,
        label: str,
        *,
        default: str,
        choices: tuple[str, ...] = (),
    ) -> str:
        """Read a free-form value with optional arrow-key choice cycling."""

        message = [("class:prompt", f"{label} ❯ ")]
        if self._prompt_session is not None:
            completer = None
            if choices:
                completer = WordCompleter(choices, ignore_case=True)
            return self._prompt_session.prompt(
                message,
                default=default,
                completer=completer,
                key_bindings=create_choice_key_bindings(choices),
            ).strip()
        value = input(f"{label} [{default}] ❯ ").strip()
        return value or default

    def _read_prompt(self) -> str | None:
        """Read the next prompt, translating terminal controls into shell state.

        Returns an empty string when Escape cancels input, and ``None`` when
        the shell should exit.
        """

        try:
            if self._prompt_session is not None:
                return self._prompt_session.prompt(
                    [("class:prompt", "\n❯ ")]
                ).strip()
            return input("\n❯ ").strip()
        except WorkflowCancelled:
            return ""
        except KeyboardInterrupt:
            if self._confirm_exit():
                self._display.notice("Goodbye.")
                return None
            self._display.warning("Exit cancelled")
            return ""
        except EOFError:
            self._display.notice("Goodbye.")
            return None

    def _confirm_exit(self) -> bool:
        """Ask for confirmation, with Yes selected by default."""

        try:
            answer = self._read_choice(
                "Exit program?",
                default="yes",
                choices=("yes", "no"),
            )
        except WorkflowCancelled:
            return False
        except (EOFError, KeyboardInterrupt):
            return True
        return answer.strip().lower() in {"", "y", "yes"}


def main() -> int:
    """Parse configuration and run the coding-agent application."""

    return CodingAgentApplication(parse_cli_arguments()).run()
