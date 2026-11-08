import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

from rich.console import Console

from coding_agent.application import (
    EXIT_SUCCESS,
    InteractiveShell,
)
from coding_agent.commands import WorkflowCancelled
from coding_agent.commands.slash_commands.model import parse_model_selection
from coding_agent.terminal import AgentIdentity, TerminalDisplay


class ModelSelectionTests(unittest.TestCase):
    def make_shell(
        self,
        *,
        model_switcher=None,
        model_lister=None,
    ) -> tuple[InteractiveShell, StringIO]:
        output = StringIO()
        display = TerminalDisplay(
            AgentIdentity(
                project_root=Path("/project"),
                provider="openai",
                model="old-model",
                session_id="thread",
                resumed=True,
            ),
            console=Console(file=output, force_terminal=False),
        )
        return (
            InteractiveShell(
                runner=Mock(),
                sessions=Mock(),
                display=display,
                model_switcher=model_switcher or Mock(),
                model_lister=model_lister,
            ),
            output,
        )

    def test_model_id_uses_the_current_provider(self) -> None:
        self.assertEqual(
            parse_model_selection("new-model", "openai"),
            ("openai", "new-model"),
        )

    def test_provider_prefix_switches_provider(self) -> None:
        self.assertEqual(
            parse_model_selection("anthropic:claude-model", "openai"),
            ("anthropic", "claude-model"),
        )

    def test_empty_model_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            parse_model_selection("mlx:", "openai")

    def test_switch_replaces_runtime_and_updates_footer(self) -> None:
        replacements: list[tuple[str, str]] = []
        output = StringIO()
        display = TerminalDisplay(
            AgentIdentity(
                project_root=Path("/project"),
                provider="openai",
                model="old-model",
                session_id="thread",
                resumed=True,
            ),
            console=Console(file=output, force_terminal=False),
        )
        shell = InteractiveShell(
            runner=object(),
            sessions=object(),
            display=display,
            model_switcher=lambda provider, model: replacements.append(
                (provider, model)
            ),
        )

        shell._commands.dispatch(
            "/model mlx:local-model",
            shell._command_context,
        )

        self.assertEqual(replacements, [("mlx", "local-model")])
        self.assertEqual(display.identity.provider, "mlx")
        self.assertEqual(display.identity.model, "local-model")
        self.assertIn("Using mlx · local-model", output.getvalue())

    def test_guided_picker_supplies_arrow_choices_at_every_step(self) -> None:
        replacements: list[tuple[str, str]] = []
        shell, _ = self.make_shell(
            model_switcher=lambda provider, model: replacements.append(
                (provider, model)
            ),
            model_lister=lambda provider: (
                "claude-alpha",
                "claude-beta",
            ),
        )
        shell._command_context.read_choice = Mock(
            side_effect=["anthropic", "claude-beta"]
        )

        shell._commands.dispatch("/model", shell._command_context)

        self.assertEqual(replacements, [("anthropic", "claude-beta")])
        self.assertEqual(
            shell._command_context.read_choice.call_args_list[0].kwargs["choices"],
            ("openai", "anthropic", "mlx"),
        )
        self.assertEqual(
            shell._command_context.read_choice.call_args_list[1].kwargs["choices"],
            ("claude-alpha", "claude-beta"),
        )

    def test_escape_cancels_the_entire_model_workflow(self) -> None:
        switcher = Mock()
        shell, output = self.make_shell(model_switcher=switcher)
        shell._command_context.read_choice = Mock(
            side_effect=WorkflowCancelled()
        )

        shell._commands.dispatch("/model", shell._command_context)

        switcher.assert_not_called()
        self.assertIn("Model selection cancelled", output.getvalue())

    def test_control_c_exits_when_default_yes_is_accepted(self) -> None:
        shell, output = self.make_shell()
        prompt_session = Mock()
        prompt_session.prompt.side_effect = KeyboardInterrupt
        shell._prompt_session = prompt_session
        shell._read_choice = Mock(return_value="yes")

        self.assertEqual(shell.run(), EXIT_SUCCESS)
        self.assertIn("Goodbye", output.getvalue())
        self.assertEqual(
            shell._read_choice.call_args.kwargs,
            {"default": "yes", "choices": ("yes", "no")},
        )


if __name__ == "__main__":
    unittest.main()
