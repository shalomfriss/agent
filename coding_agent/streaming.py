"""Streaming execution and terminal presentation."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from typing import Any, Protocol

from langchain_core.messages import AIMessageChunk, BaseMessage, ToolMessage


class StreamingAgent(Protocol):
    """The small part of a compiled agent used by the terminal runner."""

    def stream(
        self,
        input: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: list[str],
    ) -> Iterable[tuple[str, Any]]: ...


class AgentRunner:
    """Submit prompts to one session and render events as they arrive."""

    def __init__(self, agent: StreamingAgent):
        self._agent = agent

    def run(self, prompt: str, session_id: str) -> None:
        renderer = StreamPrinter()
        try:
            events = self._agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": session_id}},
                stream_mode=["messages", "updates"],
            )
            for event_type, event in events:
                self._render_event(renderer, event_type, event)
        except Exception as error:
            self._report_failure(error)
            raise
        finally:
            renderer.finish()

    @staticmethod
    def _render_event(
        renderer: StreamPrinter,
        event_type: str,
        event: Any,
    ) -> None:
        if event_type == "messages":
            message = event[0] if isinstance(event, tuple) else event
            if isinstance(message, BaseMessage):
                renderer.token(message)
        elif event_type == "updates":
            renderer.update(event)

    @staticmethod
    def _report_failure(error: Exception) -> None:
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
    """Render token chunks and tool activity once, in chronological order."""

    def __init__(self) -> None:
        self._assistant_line_open = False
        self._seen_tool_results: set[str] = set()
        self._seen_tool_calls: set[str] = set()

    def token(self, message: BaseMessage) -> None:
        """Print the text carried by one assistant token chunk."""

        if not isinstance(message, AIMessageChunk):
            return

        text = _message_text(message)
        if not text:
            return

        if not self._assistant_line_open:
            print("\nAssistant:")
            self._assistant_line_open = True
        print(text, end="", flush=True)

    def update(self, event: Any) -> None:
        """Print completed tool calls and results from one graph update."""

        for message in _find_messages(event):
            if isinstance(message, ToolMessage):
                self._print_tool_result(message)
            else:
                self._print_tool_calls(message)

    def finish(self) -> None:
        """Close a partially streamed assistant line."""

        self._close_assistant_line()

    def _print_tool_result(self, message: ToolMessage) -> None:
        identifier = message.id or repr(message)
        if identifier in self._seen_tool_results:
            return

        self._seen_tool_results.add(identifier)
        self._close_assistant_line()
        print(f"\nTool result [{message.name or 'tool'}]:")
        print(_message_text(message) or "(no output)")

    def _print_tool_calls(self, message: BaseMessage) -> None:
        for call in getattr(message, "tool_calls", None) or []:
            identifier = call.get("id") or repr(call)
            if identifier in self._seen_tool_calls:
                continue

            self._seen_tool_calls.add(identifier)
            self._close_assistant_line()
            print(f"\nTool call: {call.get('name', 'unknown')}")
            print(json.dumps(call.get("args", {}), indent=2, default=str))

    def _close_assistant_line(self) -> None:
        if self._assistant_line_open:
            print()
            self._assistant_line_open = False


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
