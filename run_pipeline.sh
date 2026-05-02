#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-ecapa_tdnn}"
CONFIG="${CONFIG:-config/config.yaml}"
DATA_ROOT="${DATA_ROOT:-data/raw}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
CPU_ONLY="${CPU_ONLY:-0}"
RESUME="${RESUME:-0}"
RESUME_PATH="${RESUME_PATH:-}"
TEST_PAIRS="${TEST_PAIRS:-data/test_pairs.txt}"
AUDIO1="${AUDIO1:-}"
AUDIO2="${AUDIO2:-}"

if [ ! -f "venv/bin/activate" ]; then
  python3 -m venv venv
fi

# shellcheck source=/dev/null
source venv/bin/activate
python -m pip install --upgrade pip

if [ "$SKIP_INSTALL" = "0" ]; then
  pip install -r requirements.txt
  if [ "$CPU_ONLY" = "0" ]; then
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio --upgrade
  fi
fi

if [ "$CPU_ONLY" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=""
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Data root not found: $DATA_ROOT" >&2
  exit 1
fi

RESUME_ARGS=()
if [ "$RESUME" = "1" ]; then
  if [ -n "$RESUME_PATH" ] && [ -f "$RESUME_PATH" ]; then
    RESUME_ARGS=(--resume "$RESUME_PATH")
  else
    MODEL_DIR="$CHECKPOINT_DIR/$MODEL"
    if [ -d "$MODEL_DIR" ]; then
      BEST="$MODEL_DIR/best_model.pt"
      LATEST=$(ls -t "$MODEL_DIR"/checkpoint_epoch_*.pt 2>/dev/null | head -n 1 || true)
      if [ -f "$BEST" ]; then
        RESUME_ARGS=(--resume "$BEST")
      elif [ -n "$LATEST" ]; then
        RESUME_ARGS=(--resume "$LATEST")
      fi
    fi
  fi
fi

python scripts/train.py --config "$CONFIG" --model "$MODEL" --data_root "$DATA_ROOT" --output_dir "$CHECKPOINT_DIR" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --learning_rate "$LEARNING_RATE" "${RESUME_ARGS[@]}"

CHECKPOINT_PATH="$CHECKPOINT_DIR/$MODEL/best_model.pt"

if [ -f "$CHECKPOINT_PATH" ]; then
  if [ -f "$TEST_PAIRS" ]; then
    python scripts/evaluate.py --config "$CONFIG" --model "$MODEL" --checkpoint "$CHECKPOINT_PATH" --test_data "$TEST_PAIRS"
  else
    echo "Test pairs file not found: $TEST_PAIRS" >&2
  fi
else
  echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
fi

if [ -n "$AUDIO1" ] && [ -n "$AUDIO2" ] && [ -f "$AUDIO1" ] && [ -f "$AUDIO2" ]; then
  python scripts/infer.py --config "$CONFIG" --model "$MODEL" --checkpoint "$CHECKPOINT_PATH" --mode verify --audio1 "$AUDIO1" --audio2 "$AUDIO2"
fi
