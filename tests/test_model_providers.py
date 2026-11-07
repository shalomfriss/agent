import unittest
from unittest.mock import patch

from coding_agent.runtime import create_model


class ModelProviderTests(unittest.TestCase):
    @patch("coding_agent.runtime.ChatOpenAI")
    def test_local_provider_uses_unauthenticated_endpoint(self, chat_openai) -> None:
        create_model("mlx", "http://localhost:8088/v1", "local-model", 0.0)

        chat_openai.assert_called_once_with(
            base_url="http://localhost:8088/v1",
            api_key="not-required",
            model="local-model",
            temperature=0.0,
            max_tokens=4096,
            max_retries=1,
            timeout=300,
        )

    @patch("coding_agent.runtime.ChatOpenAI")
    def test_openai_provider_uses_real_credentials(self, chat_openai) -> None:
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            create_model("openai", None, "openai-model", None)

        chat_openai.assert_called_once_with(
            model="openai-model",
            max_tokens=4096,
            max_retries=1,
            timeout=300,
            use_responses_api=True,
        )

    @patch("coding_agent.runtime.ChatAnthropic")
    def test_anthropic_provider_uses_anthropic_client(self, chat_anthropic) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            create_model("anthropic", None, "anthropic-model", 0.2)

        chat_anthropic.assert_called_once_with(
            model="anthropic-model",
            temperature=0.2,
            max_tokens=4096,
            max_retries=1,
            timeout=300,
        )

    def test_missing_cloud_key_is_reported(self) -> None:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "ANTHROPIC_API_KEY"):
                create_model("anthropic", None, "anthropic-model", None)


if __name__ == "__main__":
    unittest.main()
