"""Application lifecycle and interactive terminal commands."""

from __future__ import annotations

import sqlite3
import sys
from contextlib import closing

from langgraph.checkpoint.sqlite import SqliteSaver

from coding_agent.config import AgentConfig, parse_cli_arguments
from coding_agent.runtime import (
    DiagnosticResult,
    create_coding_agent,
    create_model,
    run_model_diagnostics,
)
from coding_agent.sessions import SessionManager
from coding_agent.streaming import AgentRunner
from coding_agent.workspace import Workspace


EXIT_SUCCESS = 0
EXIT_CONFIGURATION_ERROR = 1


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

        self._print_configuration(workspace)
        model = create_model(
            base_url=self._config.base_url,
            model_name=self._config.model_name,
            temperature=self._config.temperature,
        )

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
            runner = AgentRunner(agent)

            self._print_session(workspace, session_id, resumed)
            if self._config.prompt is not None:
                return self._run_once(
                    runner,
                    sessions,
                    session_id,
                    self._config.prompt,
                )

            shell = InteractiveShell(runner, sessions, session_id)
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

    def _print_configuration(self, workspace: Workspace) -> None:
        print(f"Project root: {workspace.project_root}")
        print(f"API endpoint: {self._config.base_url}")
        print(f"Model: {self._config.model_name}")

    @staticmethod
    def _print_session(
        workspace: Workspace,
        session_id: str,
        resumed: bool,
    ) -> None:
        action = "Resumed" if resumed else "Started"
        print(f"{action} session: {session_id}")
        print(f"Session database: {workspace.session_database}")
        print(f"Project memory: {workspace.memory_file}")


class InteractiveShell:
    """Read prompts and implement the agent's slash commands."""

    EXIT_COMMANDS = {"/quit", "/exit", "quit", "exit"}

    def __init__(
        self,
        runner: AgentRunner,
        sessions: SessionManager,
        session_id: str,
    ):
        self._runner = runner
        self._sessions = sessions
        self._session_id = session_id

    def run(self) -> int:
        self._print_help()
        while True:
            prompt = self._read_prompt()
            if prompt is None:
                return EXIT_SUCCESS
            if not prompt:
                continue
            if prompt.lower() in self.EXIT_COMMANDS:
                print("Goodbye.")
                return EXIT_SUCCESS
            if self._handle_command(prompt):
                continue

            try:
                self._runner.run(prompt, self._session_id)
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
        if normalized.startswith("/switch "):
            self._switch_session(prompt.split(maxsplit=1)[1])
            return True
        return False

    def _start_new_session(self) -> None:
        self._session_id, _ = self._sessions.activate(force_new=True)
        print(f"Started session: {self._session_id}")

    def _switch_session(self, requested_id: str) -> None:
        self._session_id, existed = self._sessions.activate(requested_id)
        action = "Resumed" if existed else "Started"
        print(f"{action} session: {self._session_id}")

    def _list_sessions(self) -> None:
        print("Saved sessions:")
        for session_id, updated_at in self._sessions.list_sessions():
            marker = "*" if session_id == self._session_id else " "
            print(f"{marker} {session_id}  {updated_at}")

    @staticmethod
    def _read_prompt() -> str | None:
        try:
            return input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return None

    @staticmethod
    def _print_help() -> None:
        print("\nDeep Agent is ready.")
        print("Type /quit to exit.")
        print("Type /new to start a new conversation.")
        print("Type /sessions to list saved conversations.")
        print("Type /switch SESSION_ID to change conversations.")
        print("File tools and shell commands run from the project root.")


def main() -> int:
    """Parse configuration and run the coding-agent application."""

    return CodingAgentApplication(parse_cli_arguments()).run()
