"""Command discovery, completion metadata, and dispatch."""

from __future__ import annotations

from dataclasses import dataclass

from coding_agent.commands.base import (
    CommandContext,
    CommandOutcome,
    SlashCommand,
)


@dataclass(frozen=True)
class CommandCompletion:
    """Text inserted and displayed by the slash-command completion menu."""

    completion: str
    display: str
    description: str


class CommandRegistry:
    """Index command objects once and expose dispatch and UI metadata."""

    def __init__(self, commands: tuple[SlashCommand, ...]) -> None:
        """Index primary names and aliases, rejecting ambiguous entries."""

        self.commands = commands
        self._by_name: dict[str, SlashCommand] = {}
        for command in commands:
            for name in (command.name, *command.aliases):
                if name in self._by_name:
                    raise ValueError(f"Duplicate slash command: {name}")
                self._by_name[name] = command

    def dispatch(
        self,
        prompt: str,
        context: CommandContext,
    ) -> CommandOutcome | None:
        """Execute a recognized slash command or return ``None``."""

        name, separator, remainder = prompt.partition(" ")
        command = self._by_name.get(name.lower())
        if command is None:
            return None
        argument = remainder.strip() if separator else ""
        return command.execute(context, argument)

    def completions(self) -> tuple[CommandCompletion, ...]:
        """Return completion entries for primary names and aliases."""

        entries: list[CommandCompletion] = []
        for command in self.commands:
            entries.append(
                CommandCompletion(
                    completion=(
                        f"{command.name} "
                        if command.takes_argument
                        else command.name
                    ),
                    display=command.usage,
                    description=command.description,
                )
            )
            entries.extend(
                CommandCompletion(alias, alias, command.description)
                for alias in command.aliases
            )
        return tuple(entries)

    def help_rows(self) -> tuple[tuple[str, str], ...]:
        """Return aligned usage and description rows for `/help`."""

        return tuple(
            (command.usage, command.description) for command in self.commands
        )
