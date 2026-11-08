"""Exit the interactive shell."""

from coding_agent.commands.base import CommandContext, CommandOutcome, SlashCommand


class ExitCommand(SlashCommand):
    """Handle explicit slash-command exit aliases."""

    name = "/quit"
    usage = "/quit"
    description = "Exit the agent"
    aliases = ("/exit",)

    def execute(
        self,
        context: CommandContext,
        argument: str,
    ) -> CommandOutcome:
        """Request shell termination without another confirmation prompt."""

        del context, argument
        return CommandOutcome.EXIT
