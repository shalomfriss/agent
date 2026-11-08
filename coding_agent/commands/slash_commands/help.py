"""Render help generated from registered command metadata."""

from coding_agent.commands.base import CommandContext, CommandOutcome, SlashCommand


class HelpCommand(SlashCommand):
    """Keep help output synchronized with the command registry."""

    name = "/help"
    usage = "/help"
    description = "Show commands and shortcuts"

    def execute(
        self,
        context: CommandContext,
        argument: str,
    ) -> CommandOutcome:
        """Render registry-provided usage rows and keyboard shortcuts."""

        del argument
        context.display.help(context.help_rows)
        return CommandOutcome.CONTINUE
