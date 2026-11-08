"""List project-scoped conversation threads."""

from rich.markup import escape

from coding_agent.commands.base import CommandContext, CommandOutcome, SlashCommand


class SessionsCommand(SlashCommand):
    """Render saved threads with the active thread marked."""

    name = "/sessions"
    usage = "/sessions"
    description = "List saved conversations"

    def execute(
        self,
        context: CommandContext,
        argument: str,
    ) -> CommandOutcome:
        """Read ordered session metadata and render it through Rich."""

        del argument
        context.display.console.print("\n[bold #a78bfa]Saved threads[/]")
        for session_id, updated_at in context.sessions.list_sessions():
            marker = "◆" if session_id == context.session_id else "·"
            context.display.console.print(
                f"  {marker} {escape(session_id)}  "
                f"[#64748b]{escape(updated_at)}[/]"
            )
        return CommandOutcome.CONTINUE
