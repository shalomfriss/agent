import unittest
from io import StringIO
from pathlib import Path

from rich.console import Console

from coding_agent.application import InteractiveShell, parse_model_selection
from coding_agent.terminal import AgentIdentity, TerminalDisplay


class ModelSelectionTests(unittest.TestCase):
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

        shell._switch_model("mlx:local-model")

        self.assertEqual(replacements, [("mlx", "local-model")])
        self.assertEqual(display.identity.provider, "mlx")
        self.assertEqual(display.identity.model, "local-model")
        self.assertIn("Using mlx · local-model", output.getvalue())


if __name__ == "__main__":
    unittest.main()
