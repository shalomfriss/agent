import unittest

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.shortcuts import PromptSession

from coding_agent.application import (
    SlashCommandCompleter,
    create_slash_command_key_bindings,
)


class SlashCommandCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.completer = SlashCommandCompleter()
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
                "/status",
                "/clear",
                "/help",
                "/quit",
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

    def test_escape_clears_the_entire_slash_command(self) -> None:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                completer=self.completer,
                complete_while_typing=True,
                key_bindings=create_slash_command_key_bindings(),
            )
            pipe_input.send_text("/switch unfinished\x1bregular prompt\n")

            self.assertEqual(session.prompt(), "regular prompt")

    def test_alt_enter_inserts_a_newline(self) -> None:
        with create_pipe_input() as pipe_input:
            session = PromptSession(
                input=pipe_input,
                output=DummyOutput(),
                key_bindings=create_slash_command_key_bindings(),
            )
            pipe_input.send_text("first line\x1b\rsecond line\r")

            self.assertEqual(session.prompt(), "first line\nsecond line")


if __name__ == "__main__":
    unittest.main()
