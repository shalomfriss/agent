# Coding agent architecture

The application is a project-scoped terminal agent built on Deep Agents and
LangGraph. It supports local MLX endpoints, OpenAI, and Anthropic while keeping
terminal behavior and persistence independent from the selected provider.

## Component map

| Module | Responsibility |
| --- | --- |
| `commands/` | Register commands and host shared command infrastructure. |
| `commands/slash_commands/` | Define one user-facing slash command per module. |
| `config/` | Validate sectioned JSON defaults and parse CLI overrides. |
| `workspace.py` | Validate the project and initialize `.deep-agent`. |
| `runtime.py` | Build model clients, discover models, and compile the agent. |
| `sessions.py` | Track saved threads and the active thread for each project. |
| `streaming.py` | Submit turns and translate LangGraph events into UI output. |
| `terminal.py` | Render stable header, footer, help, and status components. |
| `application.py` | Wire dependencies and coordinate interactive workflows. |

`deep_agent.py` is intentionally a thin executable entry point.

## Startup lifecycle

1. `parse_cli_arguments` produces an immutable `AgentConfig`.
2. `Workspace.prepare` validates the selected project, creates durable memory
   on first use, and ensures the session database directory exists.
3. `create_model` constructs the configured provider client.
4. Optional diagnostics verify basic chat and structured tool calls.
5. The application opens one SQLite connection for the process lifetime.
6. `SessionManager.activate` selects a requested, new, or last-active thread.
7. `create_coding_agent` compiles the Deep Agents graph with project tools,
   memory files, and a `SqliteSaver` checkpointer.
8. The application runs a one-shot prompt or enters `InteractiveShell.run`.

## Turn lifecycle

```text
user prompt
    │
    ▼
InteractiveShell
    │  prompt + active thread ID
    ▼
AgentRunner
    │  LangGraph messages/updates stream
    ▼
StreamPrinter
    ├─ assistant token output
    ├─ deduplicated tool calls
    ├─ clipped tool-result previews
    └─ completion, interruption, or failure timing
```

The full tool result remains in graph state even when the terminal preview is
clipped. After every attempted interactive turn, the session timestamp is
updated so `/sessions` remains ordered by recent use.

## Persistence model

The default `.deep-agent/sessions.sqlite3` database has two owners:

- LangGraph's `SqliteSaver` persists messages, checkpoints, and graph state by
  `thread_id`.
- `SessionManager` owns `agent_sessions` and `active_agent_sessions`, which
  store project-scoped metadata and the active-thread pointer.

Durable facts that should survive across conversation threads live in
`.deep-agent/AGENTS.md`. A project-level `AGENTS.md` is loaded separately as
repository guidance.

## Model switching

`/model` discovers model IDs through the selected provider, then rebuilds the
model client and compiled agent graph. The application deliberately reuses the
same SQLite connection and active thread ID, so the replacement agent resumes
the existing conversation checkpoint. Display identity changes only after the
replacement runtime is constructed successfully.

## Configuration model

Editable application defaults live in `coding_agent/config/defaults.json`.
The `providers`, `runtime.model`, `runtime.backend`, `prompts`, `sessions`, and
`terminal` sections keep unrelated settings separate. Prompt paths are resolved
relative to the JSON file. `coding_agent.config` loads those values into
immutable typed objects, and `AgentConfig` carries the same object through
application startup. Runtime constructors receive their prompt path, model,
and backend options explicitly, so operational values are not duplicated as
Python constants.

## Terminal control flow

- Escape raises `WorkflowCancelled` from the active prompt-toolkit workflow.
  The main prompt treats it as cleared input; nested workflows such as model
  selection unwind to the main prompt.
- Ctrl+J inserts a newline into the current prompt.
- Ctrl-C enters an explicit exit-confirmation workflow.
- Ctrl-D/EOF exits when the terminal is waiting for input.

Prompt workflows use a short Escape timeout so cancellation feels immediate
while ANSI arrow-key sequences continue to work in choice menus.

## Extension points

- Add a `SlashCommand` subclass in its own `commands/slash_commands/` module,
  then export and instantiate it in `coding_agent/commands/__init__.py`.
  Completion and `/help` metadata are derived from that registry automatically.
- Add explicit tools in `create_coding_agent`; Deep Agents' built-in file,
  shell, todo, and task tools are installed by the framework.
- Extend provider support in `coding_agent/config/defaults.json`, `create_model`,
  `list_available_models`, and the provider helpers in
  `CodingAgentApplication`.
- Add new stream event presentation in `AgentRunner._render_event` and keep
  rendering state isolated inside `StreamPrinter`.

## Security boundary

`LocalShellBackend` executes commands directly with the current user's host
permissions and inherited environment. Project rooting organizes file-tool
paths but is not a process sandbox. Run the agent only in trusted development
directories and add explicit Deep Agents permissions or interrupts before
using it in less trusted environments.
