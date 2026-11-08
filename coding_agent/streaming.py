"""Translate LangGraph stream events into a compact terminal timeline.

The runner deliberately depends on a small protocol rather than a concrete
LangGraph type. ``StreamPrinter`` handles duplicate events, spinner lifecycle,
tool summaries, output clipping, and final outcome timing.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Iterable
from typing import Any, Protocol

from langchain_core.messages import AIMessageChunk, BaseMessage, ToolMessage
from rich.console import Console
from rich.markup import escape
from rich.status import Status


ACCENT = "#a78bfa"
ACTIVE = "#2dd4bf"
MUTED = "#94a3b8"
MAX_TOOL_OUTPUT_LINES = 12
MAX_TOOL_OUTPUT_CHARACTERS = 2000


class StreamingAgent(Protocol):
    """The small part of a compiled agent used by the terminal runner."""

    def stream(
        self,
        input: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: list[str],
    ) -> Iterable[tuple[str, Any]]:
        """Yield typed message and graph-update events for one prompt."""

        ...


class AgentRunner:
    """Submit prompts to one session and render events as they arrive."""

    def __init__(
        self,
        agent: StreamingAgent,
        console: Console | None = None,
    ):
        """Bind a compiled agent and optional shared Rich console."""

        self._agent = agent
        self._console = console

    def replace_agent(self, agent: StreamingAgent) -> None:
        """Use a newly configured agent for subsequent turns."""

        self._agent = agent

    def run(self, prompt: str, session_id: str) -> None:
        """Stream one turn using ``session_id`` as the checkpoint thread.

        Exceptions are reported and then re-raised so the interactive shell
        can recover while one-shot callers still receive failure semantics.
        """

        renderer = StreamPrinter(self._console)
        renderer.thinking("Thinking")
        outcome = "interrupted"
        try:
            events = self._agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": session_id}},
                stream_mode=["messages", "updates"],
            )
            for event_type, event in events:
                self._render_event(renderer, event_type, event)
            outcome = "completed"
        except Exception as error:
            outcome = "failed"
            self._report_failure(error)
            raise
        finally:
            renderer.finish(outcome)

    @staticmethod
    def _render_event(
        renderer: StreamPrinter,
        event_type: str,
        event: Any,
    ) -> None:
        """Route one LangGraph stream event to the appropriate renderer API."""

        if event_type == "messages":
            message = event[0] if isinstance(event, tuple) else event
            if isinstance(message, BaseMessage):
                renderer.token(message)
        elif event_type == "updates":
            renderer.update(event)

    @staticmethod
    def _report_failure(error: Exception) -> None:
        """Print a concise error and an MLX hint for likely tool-call failures."""

        print("\nAgent execution failed.", file=sys.stderr)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)

        likely_tool_error = any(
            phrase in str(error).lower()
            for phrase in ("tool", "function", "422", "400", "bad request")
        )
        if likely_tool_error:
            print(
                "\nThe MLX endpoint may not support the structured tool "
                "calls required by Deep Agents.",
                file=sys.stderr,
            )


class StreamPrinter:
    """Render a compact agent timeline with visible activity states."""

    def __init__(self, console: Console | None = None) -> None:
        """Initialize per-turn display state and duplicate-event tracking."""

        self._console = console or Console(highlight=False)
        self._assistant_line_open = False
        self._seen_tool_results: set[str] = set()
        self._seen_tool_calls: set[str] = set()
        self._status: Status | None = None
        self._started_at = time.monotonic()

    def thinking(self, label: str) -> None:
        """Show an animated state while waiting for the next agent event."""

        self._stop_status()
        if self._console.is_terminal:
            self._status = self._console.status(
                f"[bold {ACCENT}]◆[/] [{MUTED}]{label}…[/]",
                spinner="dots12",
                spinner_style=ACTIVE,
            )
            self._status.start()
        else:
            self._console.print(f"\n[bold {ACCENT}]◆[/] {label}…")

    def token(self, message: BaseMessage) -> None:
        """Print the text carried by one assistant token chunk."""

        if not isinstance(message, AIMessageChunk):
            return

        text = _message_text(message)
        if not text:
            return

        self._stop_status()
        if not self._assistant_line_open:
            self._console.print(f"\n[bold {ACTIVE}]◆ Response[/]")
            self._assistant_line_open = True
        self._console.print(text, end="", markup=False, highlight=False)

    def update(self, event: Any) -> None:
        """Print completed tool calls and results from one graph update."""

        for message in _find_messages(event):
            if isinstance(message, ToolMessage):
                self._print_tool_result(message)
            else:
                self._print_tool_calls(message)

    def finish(self, outcome: str = "completed") -> None:
        """Close streamed output and report elapsed time."""

        self._stop_status()
        self._close_assistant_line()
        elapsed = time.monotonic() - self._started_at
        color = {
            "completed": MUTED,
            "interrupted": "yellow",
            "failed": "red",
        }.get(outcome, MUTED)
        self._console.print(f"[{color}]  {outcome} in {elapsed:.1f}s[/]")

    def _print_tool_result(self, message: ToolMessage) -> None:
        """Render a tool result once, then resume the thinking indicator."""

        identifier = message.id or repr(message)
        if identifier in self._seen_tool_results:
            return

        self._seen_tool_results.add(identifier)
        self._stop_status()
        self._close_assistant_line()
        output, hidden_lines = _tool_output_preview(
            _message_text(message) or "(no output)"
        )
        self._console.print(
            f"[{ACTIVE}]  └─[/] [{MUTED}]{message.name or 'tool'} finished[/]"
        )
        if output.strip():
            self._console.print(output, markup=False, highlight=False)
        if hidden_lines:
            self._console.print(
                f"[{MUTED}]     … {hidden_lines} more lines hidden[/]"
            )
        self.thinking("Reviewing results")

    def _print_tool_calls(self, message: BaseMessage) -> None:
        """Render newly observed tool calls and their compact argument summary."""

        for call in getattr(message, "tool_calls", None) or []:
            identifier = call.get("id") or repr(call)
            if identifier in self._seen_tool_calls:
                continue

            self._seen_tool_calls.add(identifier)
            self._stop_status()
            self._close_assistant_line()
            name = str(call.get("name", "unknown"))
            args = call.get("args", {})
            self._console.print(
                f"\n[bold {ACCENT}]◇ Action[/] {_tool_summary(name, args)}"
            )
            self.thinking(f"Running {name}")

    def _close_assistant_line(self) -> None:
        """Terminate streamed assistant text before block-level output."""

        if self._assistant_line_open:
            self._console.print()
            self._assistant_line_open = False

    def _stop_status(self) -> None:
        """Stop and release the active Rich spinner if one exists."""

        if self._status is not None:
            self._status.stop()
            self._status = None


def _tool_summary(name: str, args: Any) -> str:
    """Describe common coding tools in one readable line."""

    values = args if isinstance(args, dict) else {}
    if name == "execute":
        command = values.get("command", "")
        return f"[bold]{name}[/]  [#e2e8f0]$ {escape(str(command))}[/]"
    if name in {"read_file", "write_file", "edit_file"}:
        path = values.get("file_path") or values.get("path") or "file"
        verbs = {
            "read_file": "Read",
            "write_file": "Write",
            "edit_file": "Edit",
        }
        return f"[bold]{verbs[name]}[/]  [#e2e8f0]{escape(str(path))}[/]"
    if name in {"grep", "glob"}:
        query = values.get("pattern") or values.get("query") or ""
        return f"[bold]Search[/]  [#e2e8f0]{escape(str(query))}[/]"
    if name == "task":
        description = values.get("description") or values.get("prompt") or ""
        return f"[bold]Delegate[/]  [#e2e8f0]{escape(str(description))}[/]"

    compact_args = json.dumps(args, default=str, separators=(", ", ": "))
    if len(compact_args) > 160:
        compact_args = f"{compact_args[:157]}…"
    return f"[bold]{escape(name)}[/]  [#e2e8f0]{escape(compact_args)}[/]"


def _tool_output_preview(output: str) -> tuple[str, int]:
    """Keep terminal output useful without flooding the conversation view."""

    clipped = output[:MAX_TOOL_OUTPUT_CHARACTERS]
    lines = clipped.splitlines()
    visible = lines[:MAX_TOOL_OUTPUT_LINES]
    total_lines = len(output.splitlines())
    hidden_lines = max(0, total_lines - len(visible))
    if len(output) > MAX_TOOL_OUTPUT_CHARACTERS and hidden_lines == 0:
        hidden_lines = 1
    return "\n".join(f"     {line}" for line in visible), hidden_lines


def _message_text(message: BaseMessage) -> str:
    """Return displayable text from string or structured message content."""

    content = message.content
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {
            "text",
            "output_text",
        }:
            text_parts.append(str(block.get("text", "")))
    return "".join(text_parts)


def _find_messages(value: Any) -> list[BaseMessage]:
    """Recursively collect messages from a nested LangGraph update."""

    if isinstance(value, BaseMessage):
        return [value]
    if isinstance(value, dict):
        return [
            message
            for child in value.values()
            for message in _find_messages(child)
        ]
    if isinstance(value, (list, tuple)):
        return [
            message
            for child in value
            for message in _find_messages(child)
        ]
    return []
