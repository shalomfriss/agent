#!/usr/bin/env python3
"""Command-line entry point for the local coding agent.

The implementation lives in the ``coding_agent`` package. Keeping this file
small makes the execution path obvious and lets other modules be tested in
isolation.
"""

from coding_agent.application import main

# These compatibility exports preserve the original public imports.
from coding_agent.sessions import SessionManager
from coding_agent.streaming import StreamPrinter
from coding_agent.workspace import ensure_memory_file

__all__ = ["SessionManager", "StreamPrinter", "ensure_memory_file", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
