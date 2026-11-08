"""Extensible slash-command package and default registry factory."""

from coding_agent.commands.base import (
    CommandContext,
    CommandOutcome,
    ExitRequested,
    ModelLister,
    ModelSwitcher,
    WorkflowCancelled,
)
from coding_agent.commands.completion import SlashCommandCompleter
from coding_agent.commands.registry import CommandRegistry
from coding_agent.commands.slash_commands import (
    ClearCommand,
    ExitCommand,
    HelpCommand,
    ModelCommand,
    NewCommand,
    SessionsCommand,
    StatusCommand,
    SwitchCommand,
)


def create_default_registry() -> CommandRegistry:
    """Create the ordered command set used by completion, help, and dispatch."""

    return CommandRegistry(
        (
            NewCommand(),
            SessionsCommand(),
            SwitchCommand(),
            ModelCommand(),
            StatusCommand(),
            ClearCommand(),
            HelpCommand(),
            ExitCommand(),
        )
    )


__all__ = [
    "CommandContext",
    "CommandOutcome",
    "CommandRegistry",
    "ExitRequested",
    "ModelLister",
    "ModelSwitcher",
    "SlashCommandCompleter",
    "WorkflowCancelled",
    "create_default_registry",
]
