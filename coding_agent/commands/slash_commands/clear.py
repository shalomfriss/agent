"""Clear and redraw the terminal interface."""

from coding_agent.commands.base import CommandContext, CommandOutcome, SlashCommand


class ClearCommand(SlashCommand):
    """Reset terminal output while preserving runtime and session state."""

    name = "/clear"
    usage = "/clear"
    description = "Clear the terminal"

    def execute(
        self,
        context: CommandContext,
        argument: str,
    ) -> CommandOutcome:
        """Clear the console and redraw the workspace header."""

        del argument
        context.display.clear()
        return CommandOutcome.CONTINUE
