"""Start a new conversation thread."""

from coding_agent.commands.base import CommandContext, CommandOutcome, SlashCommand


class NewCommand(SlashCommand):
    """Create a fresh checkpoint thread and make it active."""

    name = "/new"
    usage = "/new"
    description = "Start a new conversation"

    def execute(
        self,
        context: CommandContext,
        argument: str,
    ) -> CommandOutcome:
        """Activate a generated session ID and refresh visible state."""

        del argument
        context.session_id, _ = context.sessions.activate(force_new=True)
        context.display.set_session(context.session_id, resumed=False)
        return CommandOutcome.CONTINUE
