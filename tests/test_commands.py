import unittest
from pathlib import Path
from unittest.mock import Mock

from coding_agent.commands import (
    CommandContext,
    CommandOutcome,
    create_default_registry,
)
from coding_agent.config import DEFAULTS
from coding_agent.terminal import AgentIdentity


class CommandRegistryTests(unittest.TestCase):
    def make_context(self) -> CommandContext:
        display = Mock()
        display.identity = AgentIdentity(
            project_root=Path("/project"),
            provider="openai",
            model="model",
            session_id="active-thread",
            resumed=True,
        )
        return CommandContext(
            sessions=Mock(),
            display=display,
            session_id="active-thread",
            model_switcher=Mock(),
            model_lister=Mock(return_value=()),
            defaults=DEFAULTS,
            read_choice=Mock(),
            confirm_exit=Mock(return_value=False),
        )

    def test_unknown_input_is_left_for_the_agent(self) -> None:
        registry = create_default_registry()

        self.assertIsNone(registry.dispatch("explain this code", self.make_context()))
        self.assertIsNone(registry.dispatch("/unknown", self.make_context()))

    def test_new_command_updates_shared_session_state(self) -> None:
        registry = create_default_registry()
        context = self.make_context()
        context.sessions.activate.return_value = ("new-thread", False)

        outcome = registry.dispatch("/new", context)

        self.assertIs(outcome, CommandOutcome.CONTINUE)
        self.assertEqual(context.session_id, "new-thread")
        context.sessions.activate.assert_called_once_with(force_new=True)
        context.display.set_session.assert_called_once_with(
            "new-thread",
            resumed=False,
        )

    def test_exit_command_terminates_the_shell(self) -> None:
        registry = create_default_registry()
        context = self.make_context()

        self.assertIs(
            registry.dispatch("/exit", context),
            CommandOutcome.EXIT,
        )
        self.assertIsNone(registry.dispatch("/quit", context))

    def test_completion_and_help_are_generated_from_registry_metadata(self) -> None:
        registry = create_default_registry()

        completion_names = [entry.completion for entry in registry.completions()]
        help_names = [usage for usage, _description in registry.help_rows()]

        self.assertIn("/model ", completion_names)
        self.assertIn("/exit", completion_names)
        self.assertNotIn("/quit", completion_names)
        self.assertIn("/model [PROVIDER:]MODEL", help_names)
        self.assertIn("/exit", help_names)

    def test_help_command_uses_registry_rows(self) -> None:
        registry = create_default_registry()
        context = self.make_context()
        context.help_rows = registry.help_rows()

        outcome = registry.dispatch("/help", context)

        self.assertIs(outcome, CommandOutcome.CONTINUE)
        context.display.help.assert_called_once_with(context.help_rows)


if __name__ == "__main__":
    unittest.main()
