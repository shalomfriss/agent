"""Concrete slash-command implementations."""

from coding_agent.commands.slash_commands.clear import ClearCommand
from coding_agent.commands.slash_commands.exit import ExitCommand
from coding_agent.commands.slash_commands.help import HelpCommand
from coding_agent.commands.slash_commands.model import ModelCommand
from coding_agent.commands.slash_commands.new import NewCommand
from coding_agent.commands.slash_commands.sessions import SessionsCommand
from coding_agent.commands.slash_commands.status import StatusCommand
from coding_agent.commands.slash_commands.switch import SwitchCommand

__all__ = [
    "ClearCommand",
    "ExitCommand",
    "HelpCommand",
    "ModelCommand",
    "NewCommand",
    "SessionsCommand",
    "StatusCommand",
    "SwitchCommand",
]
