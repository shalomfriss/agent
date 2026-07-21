import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from deep_agent import SessionManager, StreamPrinter, ensure_memory_file


class MessageState(TypedDict):
    messages: Annotated[list, add_messages]


class SessionPersistenceTests(unittest.TestCase):
    def test_memory_file_is_created_once_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            memory_file = ensure_memory_file(project)
            memory_file.write_text("keep this", encoding="utf-8")

            self.assertEqual(ensure_memory_file(project), memory_file)
            self.assertEqual(memory_file.read_text(encoding="utf-8"), "keep this")

    def test_stream_printer_renders_tokens_and_deduplicates_tools(self) -> None:
        output = StringIO()
        printer = StreamPrinter()
        tool_call = {
            "name": "execute",
            "args": {"command": "pwd"},
            "id": "call-1",
            "type": "tool_call",
        }
        tool_result = ToolMessage(
            content="/project",
            name="execute",
            tool_call_id="call-1",
            id="result-1",
        )

        with redirect_stdout(output):
            printer.token(AIMessageChunk(content="hel"))
            printer.token(AIMessageChunk(content="lo"))
            printer.update(
                {
                    "messages": [
                        AIMessage(content="", tool_calls=[tool_call])
                    ]
                }
            )
            printer.update({"messages": [tool_result]})
            printer.update({"messages": [tool_result]})
            printer.finish()

        rendered = output.getvalue()
        self.assertIn("hello", rendered)
        self.assertEqual(rendered.count("Tool call: execute"), 1)
        self.assertEqual(rendered.count("Tool result [execute]"), 1)

    def test_active_session_survives_database_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.sqlite3"
            project = Path(directory) / "project"

            first_connection = sqlite3.connect(
                database,
                check_same_thread=False,
            )
            first_manager = SessionManager(first_connection, project)
            thread_id, resumed = first_manager.activate("planning")
            first_connection.close()

            self.assertEqual(thread_id, "planning")
            self.assertFalse(resumed)

            second_connection = sqlite3.connect(
                database,
                check_same_thread=False,
            )
            second_manager = SessionManager(second_connection, project)
            thread_id, resumed = second_manager.activate()
            second_connection.close()

            self.assertEqual(thread_id, "planning")
            self.assertTrue(resumed)

    def test_checkpoint_state_survives_database_reopen(self) -> None:
        def passthrough(_state: MessageState) -> dict:
            return {}

        builder = StateGraph(MessageState)
        builder.add_node("passthrough", passthrough)
        builder.add_edge(START, "passthrough")
        config = {"configurable": {"thread_id": "persistent-thread"}}

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "sessions.sqlite3"
            first_connection = sqlite3.connect(
                database,
                check_same_thread=False,
            )
            first_graph = builder.compile(
                checkpointer=SqliteSaver(first_connection)
            )
            first_graph.invoke(
                {"messages": [HumanMessage(content="remember me")]},
                config,
            )
            first_connection.close()

            second_connection = sqlite3.connect(
                database,
                check_same_thread=False,
            )
            second_graph = builder.compile(
                checkpointer=SqliteSaver(second_connection)
            )
            restored = second_graph.get_state(config)
            second_connection.close()

            self.assertEqual(
                restored.values["messages"][0].content,
                "remember me",
            )


if __name__ == "__main__":
    unittest.main()
