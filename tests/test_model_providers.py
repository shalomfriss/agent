import unittest
from types import SimpleNamespace
from unittest.mock import patch

from coding_agent.config import DEFAULTS, ModelOptions
from coding_agent.runtime import create_model, list_available_models


class ModelProviderTests(unittest.TestCase):
    @patch("coding_agent.runtime.ChatOpenAI")
    def test_local_provider_uses_unauthenticated_endpoint(self, chat_openai) -> None:
        options = ModelOptions(max_tokens=1234, max_retries=4, timeout=45)
        create_model(
            "mlx",
            "http://localhost:8088/v1",
            "local-model",
            0.0,
            options,
        )

        chat_openai.assert_called_once_with(
            base_url="http://localhost:8088/v1",
            api_key="not-required",
            model="local-model",
            temperature=0.0,
            max_tokens=1234,
            max_retries=4,
            timeout=45,
        )

    @patch("coding_agent.runtime.ChatOpenAI")
    def test_openai_provider_uses_real_credentials(self, chat_openai) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            create_model(
                "openai",
                None,
                "openai-model",
                None,
                DEFAULTS.runtime.model,
            )

        chat_openai.assert_called_once_with(
            model="openai-model",
            max_tokens=DEFAULTS.runtime.model.max_tokens,
            max_retries=DEFAULTS.runtime.model.max_retries,
            timeout=DEFAULTS.runtime.model.timeout,
            use_responses_api=True,
        )

    @patch("coding_agent.runtime.ChatAnthropic")
    def test_anthropic_provider_uses_anthropic_client(self, chat_anthropic) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            create_model(
                "anthropic",
                None,
                "anthropic-model",
                0.2,
                DEFAULTS.runtime.model,
            )

        chat_anthropic.assert_called_once_with(
            model="anthropic-model",
            temperature=0.2,
            max_tokens=DEFAULTS.runtime.model.max_tokens,
            max_retries=DEFAULTS.runtime.model.max_retries,
            timeout=DEFAULTS.runtime.model.timeout,
        )

    def test_missing_cloud_key_is_reported(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "ANTHROPIC_API_KEY"):
                create_model(
                    "anthropic",
                    None,
                    "anthropic-model",
                    None,
                    DEFAULTS.runtime.model,
                )

    @patch("coding_agent.runtime.OpenAI")
    def test_openai_model_catalog_is_sorted_and_deduplicated(
        self,
        openai,
    ) -> None:
        openai.return_value.models.list.return_value.data = [
            SimpleNamespace(id="z-model"),
            SimpleNamespace(id="a-model"),
            SimpleNamespace(id="z-model"),
        ]

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            models = list_available_models("openai", None)

        self.assertEqual(models, ("a-model", "z-model"))
        openai.assert_called_once_with(api_key="test-key", base_url=None)

    @patch("coding_agent.runtime.Anthropic")
    def test_anthropic_model_catalog_requests_all_models(
        self,
        anthropic,
    ) -> None:
        anthropic.return_value.models.list.return_value.data = [
            SimpleNamespace(id="claude-model")
        ]

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            models = list_available_models("anthropic", None)

        self.assertEqual(models, ("claude-model",))
        anthropic.return_value.models.list.assert_called_once_with(limit=1000)


if __name__ == "__main__":
    unittest.main()
