# Repository conventions

## Slash commands

- Put every user-facing slash command in
  `coding_agent/commands/slash_commands/`, with one command per module.
- Keep shared command contracts, completion support, and registry behavior in
  `coding_agent/commands/` rather than in a concrete command module.
- Export and register new commands through `coding_agent/commands/__init__.py`.
  Command metadata is the source for both completion and `/help` output.
- Use `CommandContext` for application services instead of importing the
  interactive shell from a command.

## Configuration

- Store editable default values in `coding_agent/config/defaults.json`, grouped
  under the existing logical sections: `providers`, `runtime.model`,
  `runtime.backend`, `prompts`, `sessions`, and `terminal`. Relative prompt
  paths are resolved from the directory containing `defaults.json`.
- Keep `coding_agent/config/config.py` limited to typed loading, validation,
  path resolution, and CLI overrides. Do not add hard-coded operational
  defaults to Python modules.
- Import configuration through the stable `coding_agent.config` package API.
- Pass typed runtime options from `AgentConfig.defaults` into runtime
  constructors. Do not duplicate model token, retry, timeout, backend, session,
  or terminal defaults at their call sites.

## Keeping changes complete

- Update `coding_agent/commands/README.md` when adding or moving commands.
- Update configuration and runtime tests whenever a JSON section or typed
  configuration object changes.
