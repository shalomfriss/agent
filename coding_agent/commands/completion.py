"""Prompt-toolkit completion adapter for a command registry."""

from __future__ import annotations

from collections.abc import Iterable

from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document

from coding_agent.commands.registry import CommandRegistry


class SlashCommandCompleter(Completer):
    """Offer registered commands without completing ordinary prompt text."""

    def __init__(self, registry: CommandRegistry) -> None:
        """Use registry metadata as the single completion source."""

        self._registry = registry

    def get_completions(
        self,
        document: Document,
        complete_event: CompleteEvent,
    ) -> Iterable[Completion]:
        """Yield prefix matches only while input starts with `/`."""

        del complete_event
        prefix = document.text_before_cursor
        if not prefix.startswith("/") or " " in prefix:
            return

        normalized = prefix.lower()
        for entry in self._registry.completions():
            if entry.completion.startswith(normalized):
                yield Completion(
                    entry.completion,
                    start_position=-len(prefix),
                    display=entry.display,
                    display_meta=entry.description,
                )
