#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    StateBackend,
)
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from tools.searxng_tool import search_web
from tools.search_code import search_code


DEFAULT_BASE_URL = "http://127.0.0.1:8088/v1"
DEFAULT_MODEL = "gemma-4-e2b-it-4bit"
DEFAULT_SESSION_DB = ".deep-agent/sessions.sqlite3"


class SessionManager:
    """Track persistent conversation threads in the checkpoint database."""

    def __init__(self, connection: sqlite3.Connection, project_root: Path):
        self.connection = connection
        self.project_root = str(project_root)
        self._setup()

    def _setup(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_sessions (
                project_root TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (project_root, thread_id)
            );

            CREATE TABLE IF NOT EXISTS active_agent_sessions (
                project_root TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def activate(
        self,
        thread_id: str | None = None,
        *,
        force_new: bool = False,
    ) -> tuple[str, bool]:
        """Activate a requested, new, or most-recent thread.

        Returns the thread ID and whether it already existed.
        """

        if thread_id is not None:
            thread_id = thread_id.strip()
            if not thread_id:
                raise ValueError("Session ID cannot be empty.")

        if thread_id is None and not force_new:
            row = self.connection.execute(
                """
                SELECT thread_id
                FROM active_agent_sessions
                WHERE project_root = ?
                """,
                (self.project_root,),
            ).fetchone()
            if row is not None:
                thread_id = row[0]

        if thread_id is None:
            thread_id = str(uuid.uuid4())

        existing = self.connection.execute(
            """
            SELECT 1
            FROM agent_sessions
            WHERE project_root = ? AND thread_id = ?
            """,
            (self.project_root, thread_id),
        ).fetchone() is not None

        timestamp = self._now()
        self.connection.execute(
            """
            INSERT INTO agent_sessions (
                project_root, thread_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(project_root, thread_id)
            DO UPDATE SET updated_at = excluded.updated_at
            """,
            (self.project_root, thread_id, timestamp, timestamp),
        )
        self.connection.execute(
            """
            INSERT INTO active_agent_sessions (project_root, thread_id)
            VALUES (?, ?)
            ON CONFLICT(project_root)
            DO UPDATE SET thread_id = excluded.thread_id
            """,
            (self.project_root, thread_id),
        )
        self.connection.commit()
        return thread_id, existing

    def list_sessions(self) -> list[tuple[str, str]]:
        """Return this project's session IDs and last-used timestamps."""

        return self.connection.execute(
            """
            SELECT thread_id, updated_at
            FROM agent_sessions
            WHERE project_root = ?
            ORDER BY updated_at DESC
            """,
            (self.project_root,),
        ).fetchall()


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


@tool
def project_information() -> str:
    """Return information about the project workspace available to the agent."""
    return (
        "The project is mounted inside the agent at /workspace/. "
        "Use /workspace/ when reading, searching, creating, or editing files."
    )


def create_model(
    base_url: str,
    model_name: str,
    temperature: float,
) -> ChatOpenAI:
    """Create a LangChain client for the local OpenAI-compatible server."""

    return ChatOpenAI(
        base_url=base_url,
        api_key="not-required",
        model=model_name,
        temperature=temperature,
        max_tokens=4096,

        # Some compatible servers have problems streaming partial tool calls.
        # LangChain will disable streaming only while tools are bound.
        disable_streaming="tool_calling",

        # Useful when a server rejects unsupported OpenAI parameters.
        max_retries=1,
        timeout=300,
    )


def create_agent(
    model: ChatOpenAI,
    project_root: Path,
    checkpointer: SqliteSaver,
):
    """
    Create a Deep Agent whose real project files appear under /workspace/.

    Deep Agents' internal files remain in StateBackend instead of being
    written into the user's project.
    """

    backend = CompositeBackend(
        default=StateBackend(),
        routes={
            "/workspace/": FilesystemBackend(
                root_dir=str(project_root),
                virtual_mode=True,
            ),
        },
    )

    system_prompt = """
You are a local software-development agent running on the user's Mac.

The user's project directory is mounted at /workspace/.

Rules:
1. Inspect relevant files before proposing or making changes.
2. Use /workspace/ for all project file paths.
3. Never invent file contents.
4. Explain consequential changes clearly.
5. Prefer small, focused edits over broad rewrites.
6. Do not claim that you ran a tool unless the tool actually ran.
7. Use the multiply tool for multiplication when explicitly requested.
8. Before editing multiple files, create a concise plan using your planning tools.
9. Do not access paths outside /workspace/.
10. Ask before deleting important files or performing destructive changes.

You may use subagents for complex tasks when delegation is helpful.
""".strip()

    return create_deep_agent(
        model=model,
        tools=[
            multiply,
            project_information,
            search_web,
        ],
        backend=backend,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
    )


def print_message(message: BaseMessage) -> None:
    """Print a LangChain message in a readable form."""

    message_type = getattr(message, "type", message.__class__.__name__)
    content = getattr(message, "content", "")

    if message_type == "ai":
        print("\nAssistant:")
    elif message_type == "tool":
        tool_name = getattr(message, "name", "tool")
        print(f"\nTool result [{tool_name}]:")
    else:
        print(f"\n{message_type.capitalize()}:")

    if isinstance(content, str):
        if content.strip():
            print(content)
    else:
        print(json.dumps(content, indent=2, default=str))

    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        for call in tool_calls:
            name = call.get("name", "unknown")
            args = call.get("args", {})
            print(f"\nTool call: {name}")
            print(json.dumps(args, indent=2, default=str))


def extract_messages(value: Any) -> list[BaseMessage]:
    """Recursively extract LangChain messages from a stream update."""

    messages: list[BaseMessage] = []

    if isinstance(value, BaseMessage):
        return [value]

    if isinstance(value, dict):
        possible_messages = value.get("messages")

        if isinstance(possible_messages, list):
            messages.extend(
                item for item in possible_messages
                if isinstance(item, BaseMessage)
            )

        for child in value.values():
            if child is possible_messages:
                continue
            messages.extend(extract_messages(child))

    elif isinstance(value, (list, tuple)):
        for child in value:
            messages.extend(extract_messages(child))

    return messages


def invoke_streaming(
    agent: Any,
    prompt: str,
    thread_id: str,
) -> None:
    """Run one prompt and print tool/model updates."""

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    seen_ids: set[str] = set()

    try:
        stream = agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            },
            config=config,
            stream_mode="updates",
        )

        for update in stream:
            for message in extract_messages(update):
                identifier = getattr(message, "id", None)

                # Avoid printing the same accumulated message repeatedly.
                if identifier and identifier in seen_ids:
                    continue

                if identifier:
                    seen_ids.add(identifier)

                print_message(message)

    except Exception as exc:
        print("\nAgent execution failed.", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)

        error_text = str(exc).lower()

        if any(
            phrase in error_text
            for phrase in (
                "tool",
                "function",
                "422",
                "400",
                "bad request",
            )
        ):
            print(
                "\nThe MLX endpoint may not support OpenAI-compatible "
                "structured tool calls required by Deep Agents.",
                file=sys.stderr,
            )

        raise


def test_basic_chat(model: ChatOpenAI) -> bool:
    """Verify that the local OpenAI-compatible chat endpoint works."""

    print("Testing basic chat...")

    try:
        response = model.invoke(
            "Respond with exactly these two words: connection successful"
        )
    except Exception as exc:
        print(f"Basic chat failed: {exc}", file=sys.stderr)
        return False

    print(f"Response: {response.content}")
    return True


def test_tool_calling(model: ChatOpenAI) -> bool:
    """
    Verify that the server returns structured OpenAI-style tool calls.

    Deep Agents cannot operate correctly without this capability.
    """

    print("\nTesting structured tool calling...")

    tool_model = model.bind_tools([multiply])

    try:
        response: AIMessage = tool_model.invoke(
            "Use the multiply tool to calculate 17 times 23. "
            "Do not calculate it yourself."
        )
    except Exception as exc:
        print(f"Tool request failed: {exc}", file=sys.stderr)
        return False

    print(f"Text content: {response.content!r}")
    print(
        "Tool calls:",
        json.dumps(response.tool_calls, indent=2, default=str),
    )

    if not response.tool_calls:
        print(
            "\nNo structured tool call was returned.\n"
            "The model may have written a tool request as ordinary text, "
            "or the MLX server may not implement OpenAI tool calling.\n"
            "Basic LangChain chat may still work, but Deep Agents will not "
            "work reliably with this endpoint.",
            file=sys.stderr,
        )
        return False

    first_call = response.tool_calls[0]

    if first_call.get("name") != "multiply":
        print(
            f"Unexpected tool selected: {first_call.get('name')}",
            file=sys.stderr,
        )
        return False

    print("Structured tool calling works.")
    return True


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Deep Agents against a local MLX OpenAI endpoint."
    )

    parser.add_argument(
        "--project",
        default=".",
        help="Project directory the agent may access. Default: current directory.",
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI-compatible API URL. Default: {DEFAULT_BASE_URL}",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model ID exposed by the server. Default: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Model temperature. Default: 0",
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip basic-chat and structured-tool-call diagnostics.",
    )

    session_group = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "--session",
        help="Resume or create a session with this ID.",
    )
    session_group.add_argument(
        "--new-session",
        action="store_true",
        help="Start a new session instead of resuming the last active one.",
    )

    parser.add_argument(
        "--session-db",
        default=DEFAULT_SESSION_DB,
        help=(
            "SQLite checkpoint path, relative to the project by default. "
            f"Default: {DEFAULT_SESSION_DB}"
        ),
    )

    parser.add_argument(
        "prompt",
        nargs="*",
        help="Optional one-shot prompt. Without this, an interactive CLI starts.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    project_root = Path(args.project).expanduser().resolve()

    if not project_root.exists():
        print(
            f"Project path does not exist: {project_root}",
            file=sys.stderr,
        )
        return 1

    if not project_root.is_dir():
        print(
            f"Project path is not a directory: {project_root}",
            file=sys.stderr,
        )
        return 1

    print(f"Project root: {project_root}")
    print(f"API endpoint: {args.base_url}")
    print(f"Model: {args.model}")

    session_db = Path(args.session_db).expanduser()
    if not session_db.is_absolute():
        session_db = project_root / session_db
    session_db.parent.mkdir(parents=True, exist_ok=True)

    model = create_model(
        base_url=args.base_url,
        model_name=args.model,
        temperature=args.temperature,
    )

    if not args.skip_tests:
        if not test_basic_chat(model):
            return 1

        if not test_tool_calling(model):
            return 2

    connection = sqlite3.connect(session_db, check_same_thread=False)

    try:
        sessions = SessionManager(connection, project_root)
        thread_id, resumed = sessions.activate(
            args.session,
            force_new=args.new_session,
        )
        checkpointer = SqliteSaver(connection)
        agent = create_agent(
            model=model,
            project_root=project_root,
            checkpointer=checkpointer,
        )

        action = "Resumed" if resumed else "Started"
        print(f"{action} session: {thread_id}")
        print(f"Session database: {session_db}")

        if args.prompt:
            prompt = " ".join(args.prompt)
            invoke_streaming(agent, prompt, thread_id)
            return 0

        print("\nDeep Agent is ready.")
        print("Type /quit to exit.")
        print("Type /new to start a new conversation.")
        print("Type /sessions to list saved conversations.")
        print("Type /switch SESSION_ID to change conversations.")
        print("Files are available to the agent under /workspace/.")

        while True:
            try:
                prompt = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                return 0

            if not prompt:
                continue

            if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
                print("Goodbye.")
                return 0

            if prompt.lower() == "/new":
                thread_id, _ = sessions.activate(force_new=True)
                print(f"Started session: {thread_id}")
                continue

            if prompt.lower() == "/sessions":
                print("Saved sessions:")
                for session_id, updated_at in sessions.list_sessions():
                    marker = "*" if session_id == thread_id else " "
                    print(f"{marker} {session_id}  {updated_at}")
                continue

            if prompt.lower().startswith("/switch "):
                requested_id = prompt.split(maxsplit=1)[1]
                thread_id, existed = sessions.activate(requested_id)
                action = "Resumed" if existed else "Started"
                print(f"{action} session: {thread_id}")
                continue

            try:
                invoke_streaming(
                    agent=agent,
                    prompt=prompt,
                    thread_id=thread_id,
                )
            except Exception:
                # Keep the CLI alive so another prompt can be attempted.
                continue
    finally:
        connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
