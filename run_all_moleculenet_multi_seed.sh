#!/bin/bash

PRETRAIN_CKPT=$1
LOG_DIR="./run_all_moleculenet_logs"

DATASETS=("hiv" "bace" "bbbp")
FREEZE_VALUES=("True" "False")
NUM_GPUS=4
BASE_PORT=29100
NUM_REPEATS=5

declare -a GPU_PIDS
declare -a GPU_DATASETS
declare -a GPU_SEEDS
declare -A used_seeds
SKIP_SEEDS=(1013 51936174 293523116 561201557 590985772 1059254472)
for s in "${SKIP_SEEDS[@]}"; do
  used_seeds[$s]=1
  used_seeds[$((s+1))]=1
  used_seeds[$((s+2))]=1
  used_seeds[$((s+3))]=1
  used_seeds[$((s+4))]=1
done
NEXT_TASK_INDEX=0

log() {
  echo "$@" | tee -a "$RUN_LOG"
}

generate_unique_seed() {
  local seed
  while :; do
    seed=$((RANDOM + RANDOM * 32768))  # 약 0 ~ 10억
    local collision=false

    for offset in {0..4}; do
      if [ -n "${used_seeds[$((seed + offset))]}" ]; then
        collision=true
        break
      fi
    done

    if [ "$collision" = false ]; then
      for offset in {0..4}; do
        used_seeds[$((seed + offset))]=1
      done
      echo $seed
      return
    fi
  done
}

# (dataset, seed, freeze_pt) 조합 생성
SEED_LOG_DIRS=()
for ((i=0; i<NUM_REPEATS; i++)); do
  SEED=$(generate_unique_seed)
  for FREEZE in "${FREEZE_VALUES[@]}"; do
    SEED_LOG_DIR="${LOG_DIR}_${FREEZE}_${SEED}"
    mkdir -p "$SEED_LOG_DIR"
    : > "$SEED_LOG_DIR/run_log.log"
    SEED_LOG_DIRS+=("$SEED_LOG_DIR")
    for DATASET in "${DATASETS[@]}"; do
      TASKS+=("${DATASET},${SEED},${FREEZE}")
    done
  done
done


run_dataset_on_gpu() {
  local GPU_ID=$1
  local TASK=$2
  IFS=',' read -r DATASET SEED FREEZE <<< "$TASK"
  local PORT=$3

  local DATASET_LOG_DIR="${LOG_DIR}_${FREEZE}_${SEED}"
  local LOG_FILE="${DATASET_LOG_DIR}/${DATASET}.log"
  RUN_LOG="${DATASET_LOG_DIR}/run_log.log"

  log "[GPU ${GPU_ID}] Launching dataset '${DATASET}' (seed=${SEED}, freeze_pt=${FREEZE}) on port ${PORT} (log → ${LOG_FILE})"

  CUDA_VISIBLE_DEVICES=$GPU_ID torchrun \
    --nproc_per_node=1 \
    --master_port=$PORT \
    ft_lightning.py \
    --batch_size 256 \
    --num_workers 8 \
    --pretrain_ckpt "$PRETRAIN_CKPT" \
    --dataset_name "$DATASET" \
    --seed "$SEED" \
    --freeze_pt "$FREEZE" \
    > "$LOG_FILE" 2>&1 &

  local PID=$!
  GPU_PIDS[$GPU_ID]=$PID
  GPU_DATASETS[$GPU_ID]=$DATASET
  GPU_SEEDS[$GPU_ID]=$SEED
}

# 초기 실행
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  if [ $NEXT_TASK_INDEX -lt ${#TASKS[@]} ]; then
    TASK=${TASKS[$NEXT_TASK_INDEX]}
    PORT=$((BASE_PORT + NEXT_TASK_INDEX))
    run_dataset_on_gpu $GPU_ID "$TASK" $PORT
    ((NEXT_TASK_INDEX++))
  fi
done

# 루프 실행
while [ $NEXT_TASK_INDEX -lt ${#TASKS[@]} ]; do
  for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PID=${GPU_PIDS[$GPU_ID]}
    if ! kill -0 $PID 2>/dev/null; then
      wait $PID
      EXIT_CODE=$?
      FINISHED_DATASET=${GPU_DATASETS[$GPU_ID]}
      SEED=${GPU_SEEDS[$GPU_ID]}
      IFS=',' read -r _ _ FREEZE <<< "${TASKS[$((NEXT_TASK_INDEX-1))]}"
      LOG_FILE="${LOG_DIR}_${FREEZE}_${SEED}/${FINISHED_DATASET}.log"

      if [ $EXIT_CODE -eq 0 ]; then
        log "✅ [GPU ${GPU_ID}] dataset '${FINISHED_DATASET}' (seed=${SEED}, freeze_pt=${FREEZE}) finished successfully"
      else
        log "❌ [GPU ${GPU_ID}] dataset '${FINISHED_DATASET}' (seed=${SEED}, freeze_pt=${FREEZE}) failed (exit code $EXIT_CODE)"
        echo "[FAILED] dataset '${FINISHED_DATASET}' (seed=${SEED}, freeze_pt=${FREEZE}) with exit code $EXIT_CODE" >> "$LOG_FILE"
      fi

      if [ $NEXT_TASK_INDEX -lt ${#TASKS[@]} ]; then
        TASK=${TASKS[$NEXT_TASK_INDEX]}
        PORT=$((BASE_PORT + NEXT_TASK_INDEX))
        run_dataset_on_gpu $GPU_ID "$TASK" $PORT
        ((NEXT_TASK_INDEX++))
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
    SEED=${GPU_SEEDS[$GPU_ID]}
    IFS=',' read -r _ _ FREEZE <<< "${TASKS[$((NEXT_TASK_INDEX-1))]}"
    LOG_FILE="${LOG_DIR}_${FREEZE}_${SEED}/${FINISHED_DATASET}.log"

    if [ $EXIT_CODE -eq 0 ]; then
      log "✅ [GPU ${GPU_ID}] dataset '${FINISHED_DATASET}' (seed=${SEED}, freeze_pt=${FREEZE}) finished successfully"
    else
      log "❌ [GPU ${GPU_ID}] dataset '${FINISHED_DATASET}' (seed=${SEED}, freeze_pt=${FREEZE}) failed (exit code $EXIT_CODE)"
      echo "[FAILED] dataset '${FINISHED_DATASET}' (seed=${SEED}, freeze_pt=${FREEZE}) with exit code $EXIT_CODE" >> "$LOG_FILE"
    fi
  fi
done

log "🎉 All datasets finished."
