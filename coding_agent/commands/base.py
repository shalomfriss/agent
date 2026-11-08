"""Shared command contracts and mutable execution context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Protocol

from coding_agent.config import ApplicationDefaults
from coding_agent.sessions import SessionManager
from coding_agent.terminal import TerminalDisplay


class WorkflowCancelled(Exception):
    """Raised when Escape cancels the currently active prompt workflow."""


class ExitRequested(Exception):
    """Raised when a nested command workflow confirms program exit."""


class CommandOutcome(Enum):
    """Tell the interactive shell how to proceed after command execution."""

    CONTINUE = auto()
    EXIT = auto()


class ChoiceReader(Protocol):
    """Signature used by commands that need an interactive choice prompt."""

    def __call__(
        self,
        label: str,
        *,
        default: str,
        choices: tuple[str, ...] = (),
    ) -> str:
        """Read one value, optionally supporting a finite choice list."""

        ...


ModelSwitcher = Callable[[str, str], None]
ModelLister = Callable[[str], tuple[str, ...]]
ExitConfirmer = Callable[[], bool]


@dataclass
class CommandContext:
    """Services and mutable state available to every slash command."""

    sessions: SessionManager
    display: TerminalDisplay
    session_id: str
    model_switcher: ModelSwitcher
    model_lister: ModelLister
    defaults: ApplicationDefaults
    read_choice: ChoiceReader
    confirm_exit: ExitConfirmer
    help_rows: tuple[tuple[str, str], ...] = field(default_factory=tuple)


class SlashCommand(ABC):
    """Base class for discoverable, executable slash commands."""

    name: str
    usage: str
    description: str
    aliases: tuple[str, ...] = ()
    takes_argument: bool = False

    @abstractmethod
    def execute(
        self,
        context: CommandContext,
        argument: str,
    ) -> CommandOutcome:
        """Execute the command against the active shell context."""
