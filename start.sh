source .venv/bin/activate
lsof -ti :8088 | xargs kill -9

mlx_vlm.server \
  --model "$(pwd)/models/gemma-4-12B-it-4bit" \
  --host 127.0.0.1 \
  --port 8088