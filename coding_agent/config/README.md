# Configuration

This folder owns application defaults and their typed Python representation.

| File | Responsibility |
| --- | --- |
| `defaults.json` | Editable operational defaults grouped by concern. |
| `config.py` | Loads and validates JSON, defines typed settings, and parses CLI overrides. |
| `__init__.py` | Exposes the stable public configuration API. |

## JSON sections

| Section | Values |
| --- | --- |
| `providers` | Default model, endpoint, and temperature for each provider. |
| `runtime.model` | Model-client token limit, retry count, and request timeout. |
| `runtime.backend` | Shell command timeout. |
| `prompts` | Prompt file paths, resolved relative to `defaults.json`. |
| `sessions` | Default SQLite database path. |
| `terminal` | Interactive terminal timing such as the Escape timeout. |

Change operational defaults in `defaults.json`, not in Python call sites.
When adding a setting, update the JSON, its immutable dataclass and validation
in `config.py`, the relevant consumer, and focused tests. Other modules should
import configuration through `coding_agent.config`.
