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



sudo docker rm -f searxng

sudo docker rm -f qdrant

docker run -d \  
  --name qdrant \  
  -p 6333:6333 \  
  -p 6334:6334 \  
  -v "$PWD/qdrant_data:/qdrant/storage" \  
  qdrant/qdrant



sudo docker run \   
--network sfnet \   
-p 16000:8080 \   
-d \   
--name searxng \   
-v "./config/:/etc/searxng/" \   
-v "./data/:/var/cache/searxng/" \   
docker.io/searxng/searxng:latest

## Note: Add json ability or you will get a 403

docker container exec -it --user root searxng /bin/sh -l
cd /etc/searxng/
vi settings.yml

#add json in the format section

# restart container

docker restart searxng
