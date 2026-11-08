# Coding agent package

This package contains the application code for the project-scoped terminal
coding agent. The repository-level `deep_agent.py` file is only the executable
entry point; behavior belongs in this package.

## Modules

| Path | Responsibility |
| --- | --- |
| `application.py` | Builds dependencies and coordinates one-shot and interactive execution. |
| `commands/` | Provides slash-command contracts, registration, completion, and implementations. |
| `config/` | Loads JSON defaults, validates them, and applies command-line overrides. |
| `runtime.py` | Creates provider clients, the shell backend, tools, and the Deep Agents graph. |
| `sessions.py` | Stores conversation metadata and selects active SQLite threads. |
| `streaming.py` | Runs agent turns and renders model and tool events as they stream. |
| `terminal.py` | Renders terminal identity, status, help, notices, and the footer. |
| `workspace.py` | Validates project paths and prepares durable project state. |

Keep modules focused on these boundaries. New user-facing commands belong in
`commands/slash_commands/`, while editable operational defaults belong in
`config/defaults.json`.
