"""Model diagnostics and construction of the Deep Agents runtime."""

from __future__ import annotations

import json
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from coding_agent.workspace import MEMORY_PATH
from tools.searxng_tool import search_web


SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "system_prompt.txt"


class DiagnosticResult(IntEnum):
    """Process exit codes produced by startup model diagnostics."""

    PASSED = 0
    CHAT_FAILED = 1
    TOOL_CALLING_FAILED = 2


def create_model(
    base_url: str,
    model_name: str,
    temperature: float,
) -> ChatOpenAI:
    """Connect LangChain to the local OpenAI-compatible model server."""

    return ChatOpenAI(
        base_url=base_url,
        api_key="not-required",
        model=model_name,
        temperature=temperature,
        max_tokens=4096,
        max_retries=1,
        timeout=300,
    )


def create_coding_agent(
    model: ChatOpenAI,
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


def run_model_diagnostics(model: ChatOpenAI) -> DiagnosticResult:
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


def _test_basic_chat(model: ChatOpenAI) -> bool:
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


def _test_structured_tool_calling(model: ChatOpenAI) -> bool:
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
