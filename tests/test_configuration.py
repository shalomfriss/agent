import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from coding_agent.config import (
    DEFAULTS,
    load_defaults,
    parse_cli_arguments,
)


class ConfigurationTests(unittest.TestCase):
    def test_project_and_session_paths_are_resolved_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory).resolve()

            config = parse_cli_arguments(["-C", str(project), "--skip-tests"])

            self.assertEqual(config.project_root, project)
            self.assertEqual(
                config.session_database,
                project / DEFAULTS.sessions.database,
            )
            self.assertFalse(config.run_diagnostics)
            self.assertEqual(config.provider, "openai")
            self.assertEqual(
                config.model_name,
                DEFAULTS.providers["openai"].model,
            )
            self.assertIsNone(config.base_url)

    def test_explicit_mlx_retains_local_defaults(self) -> None:
        config = parse_cli_arguments(["--provider", "mlx"])

        self.assertEqual(config.model_name, DEFAULTS.providers["mlx"].model)
        self.assertEqual(config.base_url, DEFAULTS.providers["mlx"].base_url)
        self.assertEqual(config.temperature, 0.0)

    def test_defaults_can_be_replaced_with_a_sectioned_json_file(self) -> None:
        values = {
            "providers": {
                "openai": {
                    "model": "custom-openai",
                    "base_url": None,
                    "temperature": None,
                },
                "anthropic": {
                    "model": None,
                    "base_url": None,
                    "temperature": None,
                },
                "mlx": {
                    "model": "custom-local",
                    "base_url": "http://localhost:9999/v1",
                    "temperature": 0.3,
                },
            },
            "runtime": {
                "model": {
                    "max_tokens": 1234,
                    "max_retries": 4,
                    "timeout": 45,
                },
                "backend": {"command_timeout": 90},
            },
            "sessions": {"database": ".state/custom.sqlite3"},
            "terminal": {"escape_timeout": 0.02},
        }
        with tempfile.TemporaryDirectory() as directory:
            defaults_path = Path(directory) / "defaults.json"
            defaults_path.write_text(json.dumps(values), encoding="utf-8")

            defaults = load_defaults(defaults_path)
            config = parse_cli_arguments(["--provider", "mlx"], defaults=defaults)

        self.assertEqual(config.model_name, "custom-local")
        self.assertEqual(config.base_url, "http://localhost:9999/v1")
        self.assertEqual(config.defaults.runtime.model.max_tokens, 1234)
        self.assertEqual(config.defaults.runtime.model.max_retries, 4)
        self.assertEqual(config.defaults.runtime.model.timeout, 45)
        self.assertEqual(config.defaults.runtime.backend.command_timeout, 90)
        self.assertEqual(config.defaults.terminal.escape_timeout, 0.02)

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
