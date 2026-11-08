"""Switch the active conversation thread."""

from coding_agent.commands.base import CommandContext, CommandOutcome, SlashCommand


class SwitchCommand(SlashCommand):
    """Resume an existing ID or create it when it is not yet known."""

    name = "/switch"
    usage = "/switch SESSION_ID"
    description = "Change conversations"
    takes_argument = True

    def execute(
        self,
        context: CommandContext,
        argument: str,
    ) -> CommandOutcome:
        """Activate the requested session and update the terminal footer."""

        if not argument:
            context.display.warning("Usage: /switch SESSION_ID")
            return CommandOutcome.CONTINUE
        context.session_id, existed = context.sessions.activate(argument)
        context.display.set_session(context.session_id, resumed=existed)
        return CommandOutcome.CONTINUE
