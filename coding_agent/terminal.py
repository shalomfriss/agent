"""Shared terminal presentation for the interactive coding agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from prompt_toolkit.formatted_text import StyleAndTextTuples
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text


ACCENT = "#a78bfa"
ACTIVE = "#2dd4bf"
MUTED = "#94a3b8"


@dataclass
class AgentIdentity:
    """Information kept visible around the interactive prompt."""

    project_root: Path
    provider: str
    model: str
    session_id: str
    resumed: bool


class TerminalDisplay:
    """Render the stable chrome surrounding streamed agent output."""

    def __init__(
        self,
        identity: AgentIdentity,
        console: Console | None = None,
    ) -> None:
        self.identity = identity
        self.console = console or Console(highlight=False)

    def welcome(self) -> None:
        """Show a compact workspace card once startup is complete."""

        session_action = "resumed" if self.identity.resumed else "new"
        body = Text()
        body.append("workspace  ", style=f"bold {MUTED}")
        body.append(str(self.identity.project_root))
        body.append("\nmodel      ", style=f"bold {MUTED}")
        body.append(f"{self.identity.provider} · {self.identity.model}")
        body.append("\nthread     ", style=f"bold {MUTED}")
        body.append(self.short_session, style=ACTIVE)
        body.append(f"  {session_action}", style=MUTED)

        self.console.print(
            Panel(
                body,
                title=f"[bold {ACCENT}]◆  Orbit[/]",
                subtitle=f"[{MUTED}]type / for commands[/]",
                border_style=ACCENT,
                padding=(0, 1),
            )
        )

    @property
    def short_session(self) -> str:
        session_id = self.identity.session_id
        return session_id if len(session_id) <= 12 else f"{session_id[:12]}…"

    def set_session(self, session_id: str, resumed: bool) -> None:
        self.identity.session_id = session_id
        self.identity.resumed = resumed
        action = "Resumed" if resumed else "Started"
        self.notice(f"{action} thread {self.short_session}")

    def set_model(self, provider: str, model: str) -> None:
        self.identity.provider = provider
        self.identity.model = model
        self.notice(f"Using {provider} · {model}")

    def notice(self, message: str) -> None:
        self.console.print(f"[{ACTIVE}]◆[/] {escape(message)}")

    def warning(self, message: str) -> None:
        self.console.print(f"[yellow]◆[/] {escape(message)}")

    def status(self) -> None:
        """Print the full runtime status on demand."""

        self.console.print(f"\n[bold {ACCENT}]Runtime[/]")
        workspace = escape(str(self.identity.project_root))
        model = escape(self.identity.model)
        provider = escape(self.identity.provider)
        session_id = escape(self.identity.session_id)
        self.console.print(f"  workspace  {workspace}")
        self.console.print(
            f"  model      {provider} · {model}"
        )
        self.console.print(f"  thread     {session_id}")

    def help(self) -> None:
        """Print command and keyboard help without a large boxed overlay."""

        rows = (
            ("/new", "start a new conversation"),
            ("/sessions", "list saved conversations"),
            ("/switch ID", "change conversations"),
            ("/model", "choose a model without losing the thread"),
            ("/status", "show workspace, model, and thread"),
            ("/clear", "clear the terminal"),
            ("/help", "show this help"),
            ("/quit", "exit"),
        )
        self.console.print(f"\n[bold {ACCENT}]Commands[/]")
        for command, description in rows:
            self.console.print(
                f"  [bold {ACTIVE}]{command:<14}[/] [{MUTED}]{description}[/]"
            )
        self.console.print(
            f"\n[{MUTED}]Esc cancels the current workflow · "
            "Ctrl+J adds a line · Ctrl-C opens exit confirmation[/]"
        )

    def clear(self) -> None:
        self.console.clear()
        self.welcome()

    def toolbar(self) -> StyleAndTextTuples:
        """Return prompt-toolkit fragments for the persistent footer."""

        return [
            ("class:toolbar.label", "  model "),
            ("class:toolbar.value", self.identity.model),
            ("class:toolbar.dim", "   ◆   thread "),
            ("class:toolbar.value", self.short_session),
            ("class:toolbar.dim", "   / commands  "),
        ]
