# Slash commands

Each user-facing slash command lives in its own module. A command owns both its
metadata and behavior, so completion, `/help`, and execution cannot drift apart.

## Files

| File | Command |
| --- | --- |
| `slash_commands/new.py` | `/new` creates a fresh conversation thread. |
| `slash_commands/sessions.py` | `/sessions` lists project threads. |
| `slash_commands/switch.py` | `/switch SESSION_ID` changes the active thread. |
| `slash_commands/model.py` | `/model` discovers and switches models. |
| `slash_commands/status.py` | `/status` prints runtime identity. |
| `slash_commands/clear.py` | `/clear` redraws the terminal. |
| `slash_commands/help.py` | `/help` renders registry-derived help. |
| `slash_commands/exit.py` | `/exit` terminates the shell. |

`base.py` defines the command contract and shared execution context.
`registry.py` resolves input and produces completion/help metadata.
`completion.py` adapts registry metadata to prompt-toolkit.
`slash_commands/` contains the concrete user-facing implementations and has
its own README with the module-by-module map.

## Adding a command

1. Create a module in `slash_commands/` containing a `SlashCommand` subclass.
2. Declare `name`, `usage`, `description`, and optional aliases or argument
   support on the class.
3. Implement `execute(context, argument)` and return `CommandOutcome.CONTINUE`
   or `CommandOutcome.EXIT`.
4. Instantiate the class in `create_default_registry` in `__init__.py`.
5. Add focused behavior and registry tests.

The registry automatically exposes the new command to completion and `/help`.
Commands should use `CommandContext` rather than importing the interactive
shell; this keeps command behavior independently testable.
