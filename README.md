Coding agent powered by Deep Agents with a local MLX endpoint, OpenAI, or Anthropic.

The agent can inspect and edit files, search the repository, execute shell
commands, run tests, and search the web through the local SearXNG service.
Shell commands run directly on the host with your user permissions, so use it
only in development directories you trust.

## Run it in a project

Activate this repository's virtual environment, then launch the agent from any
directory. Like Codex, `-C` (or `--cd`) selects the working directory:

```shell
/Users/shalomfriss/repos/mlx/.venv/bin/python \
  /Users/shalomfriss/repos/mlx/deep_agent.py \
  -C /path/to/project \
  --skip-tests
```

The default project is the current directory. A one-shot prompt can be placed
after the options:

```shell
./deep_agent.py -C /path/to/project --skip-tests \
  "Find the failing tests and fix them"
```

From this repository, `./start_agent.sh -C /path/to/project --skip-tests`
provides the same launcher and forwards all CLI options.

The built-in file tools treat the selected directory as `/`, and every shell
command starts there. The default provider is OpenAI with model `gpt-6-luna`.
Use `--provider mlx` for the local model endpoint at
`http://127.0.0.1:8088/v1`; override that endpoint with `--base-url`.

## Choose a model provider

OpenAI is selected by default. Pass `--model` to override its default model,
or choose `--provider anthropic` with a model ID. Set the matching API key in
your environment before starting the agent:

```shell
export OPENAI_API_KEY="your-openai-key"
./deep_agent.py -C /path/to/project

export ANTHROPIC_API_KEY="your-anthropic-key"
./deep_agent.py --provider anthropic --model YOUR_ANTHROPIC_MODEL -C /path/to/project
```

Cloud providers do not use the local MLX endpoint. `--base-url` can point the
OpenAI provider to a compatible endpoint; it is unavailable with Anthropic.
`--temperature` is optional for cloud models and is passed only when specified.
The startup chat and tool diagnostics call the chosen provider unless you pass
`--skip-tests`.

## Code structure

`deep_agent.py` is intentionally only an executable entry point. Each module
in `coding_agent/` owns one concern:

| Module | Responsibility |
| --- | --- |
| `config.py` | Parse CLI options into typed configuration. |
| `workspace.py` | Validate project paths and initialize durable memory. |
| `sessions.py` | Track active and saved SQLite conversation threads. |
| `runtime.py` | Construct the model, tools, backend, and Deep Agent graph. |
| `streaming.py` | Run prompts and render streamed model/tool events. |
| `application.py` | Coordinate startup, one-shot mode, and the interactive CLI. |

## Persistent sessions

`deep_agent.py` streams response tokens and tool activity as they happen. It
stores conversation checkpoints in
`.deep-agent/sessions.sqlite3` inside the selected project. On startup, the
last active conversation is resumed automatically, including its message and
agent state.

```shell
python deep_agent.py --skip-tests
python deep_agent.py --session feature-planning --skip-tests
python deep_agent.py --new-session --skip-tests
```

Inside the interactive CLI, use `/new` to start a fresh conversation,
`/sessions` to list saved conversations, and `/switch SESSION_ID` to resume a
specific one. Typing `/` opens the command menu; `/status`, `/clear`, and
`/help` expose runtime details and terminal controls. The footer keeps the
active model and thread visible, while animated thinking and tool states show
what the agent is doing. Press Escape to immediately clear the prompt or cancel
the active workflow, and Ctrl+J for multiline prompts. Ctrl-C asks for
confirmation before exiting, with Yes selected by default. Use
`--session-db PATH` to store the SQLite database elsewhere.

Use `/model` for the guided provider and model picker. The up and down arrows
cycle through every provider and every model reported by that provider. Escape
cancels the picker at either step. Changing models rebuilds the agent runtime
but keeps the current conversation thread. For a direct switch, use
`/model MODEL_ID` with the current provider or
`/model PROVIDER:MODEL_ID`, such as `/model mlx:models/local-model`.

Durable cross-session context lives in `.deep-agent/AGENTS.md`. The agent may
record stable project facts and explicit preferences there; ordinary dialogue
history remains in the selected SQLite session. If the project already has a
top-level `AGENTS.md`, it is also loaded as project context.

download your models into the models directory and update the deep_agent.py file to reflect that


sudo docker rm -f searxng
sudo docker rm -f qdrant

cd /Users/shalomfriss/repos/mlx
  mkdir -p qdrant_data data

  docker run -d \
    --name qdrant \
    --restart unless-stopped \
    -p 6333:6333 \
    -p 6334:6334 \
    -v "$PWD/qdrant_data:/qdrant/storage" \
    qdrant/qdrant

  docker run -d \
    --name searxng \
    --restart unless-stopped \
    -p 16000:8080 \
    -v "$PWD/config:/etc/searxng" \
    -v "$PWD/data:/var/cache/searxng" \
    docker.io/searxng/searxng:latest

  No sfnet Docker network is necessary—the Python code accesses both through localhost:

  - SearXNG: http://127.0.0.1:16000
  - Qdrant: http://127.0.0.1:6333

  Verify them:

  curl "http://127.0.0.1:16000/search?q=test&format=json"
  curl "http://127.0.0.1:6333/healthz"

  JSON output is already enabled in config/settings.yml.

  For subsequent runs:

  docker start qdrant searxng

  To inspect or stop them:

  docker logs -f searxng
  docker logs -f qdrant

  docker stop searxng qdrant

## Note: Add json ability or you will get a 403

docker container exec -it --user root searxng /bin/sh -l
cd /etc/searxng/
vi settings.yml

#add json in the format section

# restart container

docker restart searxng




Implemented a full directory-scoped coding agent.

  Key additions:

  - Project-root file editing, search, shell execution, tests, and task tools.
  - Token-level response streaming with deduplicated tool activity.
  - Persistent SQLite conversation sessions.
  - Durable cross-session memory in .deep-agent/AGENTS.md.
  - Codex-style directory selection: -C, --cd, or --project.
  - Updated coding-agent system prompt and launcher.
  - Expanded documentation and tests.

  Run it with:

  ./start_agent.sh -C /path/to/project --skip-tests

  Or one-shot:

  ./deep_agent.py -C /path/to/project --skip-tests "Fix the failing tests"
