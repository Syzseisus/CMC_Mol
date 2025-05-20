#!/bin/bash

PRETRAIN_CKPT=$1
LOG_DIR="./run_all_moleculenet_logs"

DATASETS=(hiv sider clintox bace muv tox21 bbbp toxcast)
NUM_GPUS=4
BASE_PORT=29100

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

declare -a GPU_PIDS
declare -a GPU_DATASETS
NEXT_DATA_INDEX=0

log() {
  echo "$@" | tee -a "$RUN_LOG"
}


run_dataset_on_gpu() {
  local GPU_ID=$1
  local DATASET=$2
  local PORT=$3

  local LOG_FILE="${LOG_DIR}/${DATASET}.log"
  RUN_LOG="${LOG_DIR}/run_log.log"

  log "[GPU ${GPU_ID}] Launching dataset '${DATASET}' on port ${PORT} (log → ${LOG_FILE})"

  CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=1 --master_port=$PORT \
    ft_lightning.py \
    --pretrain_ckpt "$PRETRAIN_CKPT" \
    --dataset_name "$DATASET" \
    > "$LOG_FILE" 2>&1 &

  local PID=$!
  GPU_PIDS[$GPU_ID]=$PID
  GPU_DATASETS[$GPU_ID]=$DATASET
}

# 초기 실행
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  if [ $NEXT_DATA_INDEX -lt ${#DATASETS[@]} ]; then
    DATASET=${DATASETS[$NEXT_DATA_INDEX]}
    PORT=$((BASE_PORT + NEXT_DATA_INDEX))
    run_dataset_on_gpu $GPU_ID "$DATASET" $PORT
    ((NEXT_DATA_INDEX++))
  fi
done

# 루프 실행
while [ $NEXT_DATA_INDEX -lt ${#DATASETS[@]} ]; do
  for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PID=${GPU_PIDS[$GPU_ID]}
    if ! kill -0 $PID 2>/dev/null; then
      wait $PID
      EXIT_CODE=$?
      FINISHED_DATASET=${GPU_DATASETS[$GPU_ID]}
      LOG_FILE="${LOG_DIR}/${FINISHED_DATASET}.log"

      if [ $EXIT_CODE -eq 0 ]; then
        log "✅ [GPU ${GPU_ID}] dataset '${FINISHED_DATASET}' finished successfully"
      else
        log "❌ [GPU ${GPU_ID}] dataset '${FINISHED_DATASET}' failed (exit code $EXIT_CODE)"
        echo "[FAILED] dataset '${FINISHED_DATASET}' with exit code $EXIT_CODE" >> "$LOG_FILE"
      fi

      if [ $NEXT_DATA_INDEX -lt ${#DATASETS[@]} ]; then
        DATASET=${DATASETS[$NEXT_DATA_INDEX]}
        PORT=$((BASE_PORT + NEXT_DATA_INDEX))
        run_dataset_on_gpu $GPU_ID "$DATASET" $PORT
        ((NEXT_DATA_INDEX++))
      fi
    fi
  done
  sleep 1
done

# 마지막 정리
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  PID=${GPU_PIDS[$GPU_ID]}
  if [ -n "$PID" ] && kill -0 $PID 2>/dev/null; then
    wait $PID
    EXIT_CODE=$?
    FINISHED_DATASET=${GPU_DATASETS[$GPU_ID]}
    LOG_FILE="${LOG_DIR}/${FINISHED_DATASET}.log"

    if [ $EXIT_CODE -eq 0 ]; then
      log "✅ [GPU ${GPU_ID}] dataset '${FINISHED_DATASET}' finished successfully"
    else
      log "❌ [GPU ${GPU_ID}] dataset '${FINISHED_DATASET}' failed (exit code $EXIT_CODE)"
      echo "[FAILED] dataset '${FINISHED_DATASET}' with exit code $EXIT_CODE" >> "$LOG_FILE"
    fi
  fi
done

log "🎉 All datasets finished."
