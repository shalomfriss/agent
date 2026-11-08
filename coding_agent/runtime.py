"""Construct provider clients and the project-scoped Deep Agents graph.

The runtime combines a chat model, unrestricted local project backend,
built-in Deep Agents tools, durable memory files, and a LangGraph checkpointer.
Provider discovery lives here so interactive model switching uses the same
credentials and endpoints as initial startup.
"""

from __future__ import annotations

import json
import os
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from openai import OpenAI

from coding_agent.workspace import MEMORY_PATH
from tools.searxng_tool import search_web


SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "system_prompt.txt"


class DiagnosticResult(IntEnum):
    """Process exit codes produced by startup model diagnostics."""

    PASSED = 0
    CHAT_FAILED = 1
    TOOL_CALLING_FAILED = 2


def create_model(
    provider: str,
    base_url: str | None,
    model_name: str,
    temperature: float | None,
) -> BaseChatModel:
    """Construct the selected provider's LangChain chat model.

    MLX uses the OpenAI-compatible client without authentication. Cloud
    providers require their standard environment variables and fail early with
    a readable configuration error when credentials are absent.
    """

    options: dict[str, Any] = {
        "model": model_name,
        "max_tokens": 4096,
        "max_retries": 1,
        "timeout": 300,
    }
    if temperature is not None:
        options["temperature"] = temperature

    if provider == "mlx":
        return ChatOpenAI(base_url=base_url, api_key="not-required", **options)
    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is required with --provider openai")
        if base_url:
            options["base_url"] = base_url
        else:
            options["use_responses_api"] = True
        return ChatOpenAI(**options)
    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY is required with --provider anthropic")
        return ChatAnthropic(**options)
    raise ValueError(f"Unknown model provider: {provider}")


def list_available_models(
    provider: str,
    base_url: str | None,
) -> tuple[str, ...]:
    """Return sorted, unique model IDs exposed by a provider's models API.

    Args:
        provider: One of ``mlx``, ``openai``, or ``anthropic``.
        base_url: OpenAI-compatible endpoint for MLX or a custom OpenAI server.

    Raises:
        ValueError: If the provider is unknown, a required endpoint is absent,
            or the corresponding cloud credential is unavailable.
    """

    if provider == "mlx":
        if not base_url:
            raise ValueError("An MLX base URL is required to list models")
        client = OpenAI(base_url=base_url, api_key="not-required")
        models = client.models.list().data
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required with --provider openai")
        client = OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list().data
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required with --provider anthropic"
            )
        models = Anthropic(api_key=api_key).models.list(limit=1000).data
    else:
        raise ValueError(f"Unknown model provider: {provider}")

    return tuple(sorted({model.id for model in models}, key=str.casefold))


def create_coding_agent(
    model: BaseChatModel,
    project_root: Path,
    checkpointer: SqliteSaver,
) -> Any:
    """Build an agent with project-rooted files, shell, memory, and tools.

    ``LocalShellBackend`` deliberately grants unrestricted local command
    execution. Its file API treats the project as ``/`` and its shell starts
    every command in the project directory.
    """

    backend = LocalShellBackend(
        root_dir=project_root,
        virtual_mode=True,
        timeout=300,
        inherit_env=True,
    )
    system_prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    return create_deep_agent(
        model=model,
        tools=[project_information, search_web],
        backend=backend,
        system_prompt=system_prompt,
        memory=[f"/{MEMORY_PATH.as_posix()}", "/AGENTS.md"],
        checkpointer=checkpointer,
    )


def run_model_diagnostics(model: BaseChatModel) -> DiagnosticResult:
    """Confirm chat and structured tool calls, returning a precise status."""

    if not _test_basic_chat(model):
        return DiagnosticResult.CHAT_FAILED
    if not _test_structured_tool_calling(model):
        return DiagnosticResult.TOOL_CALLING_FAILED
    return DiagnosticResult.PASSED


@tool
def project_information() -> str:
    """Describe how file paths and shell commands map to the current project."""

    return (
        "The filesystem root and shell working directory are the selected "
        "project directory. Use paths relative to that root."
    )


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers; used only to test structured tool calls."""

    return a * b


def _test_basic_chat(model: BaseChatModel) -> bool:
    """Verify that the configured endpoint can complete a minimal chat turn."""

    print("Testing basic chat...")
    try:
        response = model.invoke(
            "Respond with exactly these two words: connection successful"
        )
    except Exception as error:
        print(f"Basic chat failed: {error}", file=sys.stderr)
        return False

    print(f"Response: {response.content}")
    return True


def _test_structured_tool_calling(model: BaseChatModel) -> bool:
    """Verify that the endpoint emits structured calls required by the agent."""

    print("\nTesting structured tool calling...")
    tool_model = model.bind_tools([multiply])

    try:
        response: AIMessage = tool_model.invoke(
            "Use the multiply tool to calculate 17 times 23. "
            "Do not calculate it yourself."
        )
    except Exception as error:
        print(f"Tool request failed: {error}", file=sys.stderr)
        return False

    print(f"Text content: {response.content!r}")
    print("Tool calls:", json.dumps(response.tool_calls, indent=2, default=str))

    if not response.tool_calls:
        print(
            "\nNo structured tool call was returned. The MLX server must "
            "support OpenAI-compatible tool calls for the agent to work.",
            file=sys.stderr,
        )
        return False

    selected_tool = response.tool_calls[0].get("name")
    if selected_tool != "multiply":
        print(f"Unexpected tool selected: {selected_tool}", file=sys.stderr)
        return False

    print("Structured tool calling works.")
    return True
