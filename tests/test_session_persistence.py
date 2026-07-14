import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from deep_agent import SessionManager


class MessageState(TypedDict):
    messages: Annotated[list, add_messages]


class SessionPersistenceTests(unittest.TestCase):
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
