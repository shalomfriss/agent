from pathlib import Path
import subprocess

from langchain_core.tools import tool


PROJECT_ROOT = Path(
    "/Users/shalomfriss/repos/my-project"
).resolve()


@tool
def search_code(
    query: str,
    file_glob: str | None = None,
    max_results: int = 100,
) -> str:
    """
    Search the codebase for exact text or regular-expression matches.

    Use this for symbols, class names, functions, imports, API paths,
    error messages, configuration values, and other exact code searches.
    """
    command = [
        "rg",
        "--line-number",
        "--column",
        "--no-heading",
        "--hidden",
        "--glob",
        "!.git/**",
    ]

    if file_glob:
        command.extend(["--glob", file_glob])

    command.extend(["--", query, str(PROJECT_ROOT)])

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "Code search timed out."

    if result.returncode not in (0, 1):
        return f"Code search failed: {result.stderr.strip()}"

    lines = result.stdout.splitlines()

    if not lines:
        return "No code matches found."

    truncated = lines[:max_results]

    return "\n".join(truncated)