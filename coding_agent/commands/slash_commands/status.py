"""Display active runtime information."""

from coding_agent.commands.base import CommandContext, CommandOutcome, SlashCommand


class StatusCommand(SlashCommand):
    """Show the active project, provider, model, and thread."""

    name = "/status"
    usage = "/status"
    description = "Show model and session details"

    def execute(
        self,
        context: CommandContext,
        argument: str,
    ) -> CommandOutcome:
        """Delegate runtime rendering to the shared terminal display."""

        del argument
        context.display.status()
        return CommandOutcome.CONTINUE
