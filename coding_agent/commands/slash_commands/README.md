# Slash-command implementations

This folder contains the concrete commands users invoke from the interactive
prompt. Each command lives in its own module and subclasses `SlashCommand`.

| Module | Behavior |
| --- | --- |
| `new.py` | Starts a new conversation thread. |
| `sessions.py` | Lists saved conversation threads. |
| `switch.py` | Changes the active conversation thread. |
| `model.py` | Discovers providers and models and switches the active model. |
| `status.py` | Displays the current project, provider, model, and session. |
| `clear.py` | Clears and redraws the terminal interface. |
| `help.py` | Renders help generated from registry metadata. |
| `exit.py` | Implements `/quit` and `/exit`. |
| `__init__.py` | Exports concrete command classes to the parent package. |

When adding a command, create one module here, declare its completion/help
metadata on the class, export it from this folder's `__init__.py`, and register
an instance in `coding_agent/commands/__init__.py`.

Shared command infrastructure does not belong here. Put execution contracts,
registry behavior, and completion integration in the parent `commands/`
folder.
