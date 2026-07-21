"""Project-directory validation and persistent file setup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coding_agent.config import AgentConfig


MEMORY_PATH = Path(".deep-agent/AGENTS.md")
INITIAL_MEMORY = """# Coding Agent Memory

Store durable project facts and user preferences here as they are learned.
Keep entries concise and never store credentials or transient task details.
"""


@dataclass(frozen=True)
class Workspace:
    """Validated paths owned by one project-scoped agent invocation."""

    project_root: Path
    session_database: Path
    memory_file: Path

    @classmethod
    def prepare(cls, config: AgentConfig) -> Workspace:
        """Validate the project and initialize its private agent directory."""

        _validate_project_directory(config.project_root)
        memory_file = ensure_memory_file(config.project_root)
        config.session_database.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            project_root=config.project_root,
            session_database=config.session_database,
            memory_file=memory_file,
        )


def ensure_memory_file(project_root: Path) -> Path:
    """Create durable project memory on first use without overwriting it."""

    memory_file = project_root / MEMORY_PATH
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    if not memory_file.exists():
        memory_file.write_text(INITIAL_MEMORY, encoding="utf-8")
    return memory_file


def _validate_project_directory(project_root: Path) -> None:
    if not project_root.exists():
        raise ValueError(f"Project path does not exist: {project_root}")
    if not project_root.is_dir():
        raise ValueError(f"Project path is not a directory: {project_root}")
