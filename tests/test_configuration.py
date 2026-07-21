import tempfile
import unittest
from pathlib import Path

from coding_agent.config import DEFAULT_SESSION_DB, parse_cli_arguments


class ConfigurationTests(unittest.TestCase):
    def test_project_and_session_paths_are_resolved_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve()

            config = parse_cli_arguments(["-C", str(project), "--skip-tests"])

            self.assertEqual(config.project_root, project)
            self.assertEqual(config.session_database, project / DEFAULT_SESSION_DB)
            self.assertFalse(config.run_diagnostics)

    def test_remaining_arguments_form_the_one_shot_prompt(self) -> None:
        config = parse_cli_arguments(["fix", "the", "tests"])

        self.assertEqual(config.prompt, "fix the tests")


if __name__ == "__main__":
    unittest.main()
