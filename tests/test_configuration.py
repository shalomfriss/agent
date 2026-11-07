import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from coding_agent.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_SESSION_DB,
    parse_cli_arguments,
)


class ConfigurationTests(unittest.TestCase):
    def test_project_and_session_paths_are_resolved_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve()

            config = parse_cli_arguments(["-C", str(project), "--skip-tests"])

            self.assertEqual(config.project_root, project)
            self.assertEqual(config.session_database, project / DEFAULT_SESSION_DB)
            self.assertFalse(config.run_diagnostics)
            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.model_name, DEFAULT_OPENAI_MODEL)
            self.assertIsNone(config.base_url)

    def test_explicit_mlx_retains_local_defaults(self) -> None:
        config = parse_cli_arguments(["--provider", "mlx"])

        self.assertEqual(config.model_name, DEFAULT_MODEL)
        self.assertEqual(config.base_url, DEFAULT_BASE_URL)
        self.assertEqual(config.temperature, 0.0)

    def test_remaining_arguments_form_the_one_shot_prompt(self) -> None:
        config = parse_cli_arguments(["fix", "the", "tests"])

        self.assertEqual(config.prompt, "fix the tests")

    def test_cloud_providers_use_explicit_models_and_no_local_endpoint(self) -> None:
        for provider in ("openai", "anthropic"):
            with self.subTest(provider=provider):
                config = parse_cli_arguments(
                    ["--provider", provider, "--model", "selected-model"]
                )
                self.assertEqual(config.provider, provider)
                self.assertEqual(config.model_name, "selected-model")
                self.assertIsNone(config.base_url)
                self.assertIsNone(config.temperature)

    def test_cloud_provider_requires_model(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parse_cli_arguments(["--provider", "anthropic"])
        self.assertEqual(error.exception.code, 2)

    def test_anthropic_rejects_openai_compatible_endpoint(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parse_cli_arguments(
                    [
                        "--provider", "anthropic", "--model", "claude",
                        "--base-url", "http://localhost",
                    ]
                )
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
