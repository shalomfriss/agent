source .venv/bin/activate
lsof -ti :8088 | xargs kill -9

# mlx_vlm.server \
#   --model "$(pwd)/models/gemma-4-12B-it-4bit" \
#   --host 127.0.0.1 \
#   --max-tokens 512 \
#   --port 8088

# python -m mlx_vlm.server \
#   --model "$(pwd)/models/gemma-4-12B-it-4bit" \
#   --host 127.0.0.1 \
#   --port 8088 \
#   --kv-bits 4 \
#   --kv-quant-scheme turboquant


# APC_ENABLED=1 \
# APC_DISK_PATH="$HOME/.cache/mlx_apc" \
# APC_DISK_MAX_GB=15 \
# python -m mlx_vlm.server \
#   --model "$(pwd)/models/gemma-4-12B-it-4bit" \
#   --host 127.0.0.1 \
#   --port 8088 \
#   --kv-bits 4 \
#   --kv-quant-scheme turboquant

mlx_vlm.server \
  --model "$(pwd)/models/gemma-4-e2b-it-4bit" \
  --host 127.0.0.1 \
  --port 8088