source .venv/bin/activate
lsof -ti :8088 | xargs kill -9

sudo docker rm -f searxng
sudo docker rm -f qdrant

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

mlx_vlm.server \
  --model "$(pwd)/models/gemma-4-e2b-it-4bit" \
  --host 127.0.0.1 \
  --port 8088
  