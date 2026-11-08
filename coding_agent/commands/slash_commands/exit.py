"""Exit the interactive shell."""

from coding_agent.commands.base import CommandContext, CommandOutcome, SlashCommand


class ExitCommand(SlashCommand):
    """Exit the interactive shell from the explicit slash command."""

    name = "/exit"
    usage = "/exit"
    description = "Exit the agent"

    def execute(
        self,
        context: CommandContext,
        argument: str,
    ) -> CommandOutcome:
        """Request shell termination without another confirmation prompt."""

        del context, argument
        return CommandOutcome.EXIT
