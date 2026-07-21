Deep agent with SearXNG, Qdrant...

## Persistent sessions

`deep_agent.py` stores conversation checkpoints in
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
specific one. Use `--session-db PATH` to store the SQLite database elsewhere.

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
