import unittest
from threading import Timer

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.shortcuts import PromptSession

from coding_agent.application import (
    create_choice_key_bindings,
    create_slash_command_key_bindings,
)
from coding_agent.commands import (
    SlashCommandCompleter,
    WorkflowCancelled,
    create_default_registry,
)


class SlashCommandCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.completer = SlashCommandCompleter(create_default_registry())
        self.event = CompleteEvent(completion_requested=False)

    def completions_for(self, text: str):
        return list(
            self.completer.get_completions(Document(text), self.event)
        )

    def test_slash_lists_commands_with_descriptions(self) -> None:
        completions = self.completions_for("/")

        self.assertEqual(
            [completion.text for completion in completions],
            [
                "/new",
                "/sessions",
                "/switch ",
                "/model ",
                "/status",
                "/clear",
                "/help",
                "/exit",
            ],
        )
        self.assertEqual(
            completions[0].display_meta_text,
            "Start a new conversation",
        )

    def test_partial_command_filters_and_inserts_argument_space(self) -> None:
        completions = self.completions_for("/sw")

        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0].text, "/switch ")
        self.assertEqual(completions[0].start_position, -3)

    def test_regular_prompts_and_command_arguments_have_no_completions(self) -> None:
        self.assertEqual(self.completions_for("fix the tests"), [])
        self.assertEqual(self.completions_for("/switch feature"), [])

    def test_escape_clears_the_entire_input(self) -> None:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                completer=self.completer,
                complete_while_typing=True,
                key_bindings=create_slash_command_key_bindings(),
            )
            session.app.ttimeoutlen = 0.01
            pipe_input.send_text("unfinished prompt\x1b")
            timer = Timer(
                0.05,
                lambda: pipe_input.send_text("regular prompt\r"),
            )
            timer.start()

            try:
                with self.assertRaises(WorkflowCancelled):
                    session.prompt()
                self.assertEqual(session.prompt(), "regular prompt")
            finally:
                timer.join()

    def test_control_j_inserts_a_newline(self) -> None:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=create_slash_command_key_bindings(),
            )
            pipe_input.send_text("first line\nsecond line\r")

            self.assertEqual(session.prompt(), "first line\nsecond line")

    def test_arrow_keys_cycle_through_choices(self) -> None:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=create_choice_key_bindings(
                    ("openai", "anthropic", "mlx")
                ),
            )
            pipe_input.send_text("\x1b[B\r")

            self.assertEqual(session.prompt(default="openai"), "anthropic")

    def test_escape_cancels_a_choice_workflow(self) -> None:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=create_choice_key_bindings(("yes", "no")),
            )
            session.app.ttimeoutlen = 0.01
            pipe_input.send_text("\x1b")

            with self.assertRaises(WorkflowCancelled):
                session.prompt(default="yes")


if __name__ == "__main__":
    unittest.main()
