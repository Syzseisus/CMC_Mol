#!/bin/bash

TEST_MODE=false

# ========= 기본 세팅 =========

BASE_LOG_DIR="./run_gs_logs"
TIMESTAMP=$(TZ='Asia/Seoul' date "+%Y%m%d_%H%M%S")
LOG_DIR="${BASE_LOG_DIR}_${TIMESTAMP}"
JSON_DIR="${LOG_DIR}/results_summary"

NUM_GPUS=4
BASE_PORT=29100

mkdir -p "$LOG_DIR"
mkdir -p "$JSON_DIR"

declare -a GPU_PIDS
declare -a GPU_TASKS
declare -a GPU_PORTS
NEXT_TASK_INDEX=0

# ========= TASKS 정의 =========

declare -A TASKS_CKPT_MAP

# ckpt 경로
CKPT_A_PATH="../ckpt/pretrained/cmc_atom/aug_lambda_1/checkpoints/last.ckpt"
CKPT_B_PATH="random_init"

GS_KEYS=(full_ft zero_shot fusion_cls)

# 1. Gap 확인하기
## 1-1. full finetuning
TASKS_ckpt_a=(
  "--full_ft --dataset_name hiv"
  "--full_ft --dataset_name muv"
  "--full_ft --dataset_name sider --batch_size 128"
  "--full_ft --dataset_name clintox"
  "--full_ft --dataset_name bace"
  "--full_ft --dataset_name tox21"
  "--full_ft --dataset_name bbbp"
  "--full_ft --dataset_name toxcast"
)
## 1-2. linear probing
TASKS_ckpt_a+=(
  "--dataset_name hiv"
  "--dataset_name muv"
  "--dataset_name sider --batch_size 128"
  "--dataset_name clintox"
  "--dataset_name bace"
  "--dataset_name tox21"
  "--dataset_name bbbp"
  "--dataset_name toxcast"
)

# 2. Gain 확인하기
## 2-1. Zero-shot과 비교하기
### 2-1-1. zero shot
TASKS_ckpt_a+=(
  "--full_ft --zero_shot --dataset_name hiv"
  "--full_ft --zero_shot --dataset_name muv"
  "--full_ft --zero_shot --dataset_name sider --batch_size 128"
  "--full_ft --zero_shot --dataset_name clintox"
  "--full_ft --zero_shot --dataset_name bace"
  "--full_ft --zero_shot --dataset_name tox21"
  "--full_ft --zero_shot --dataset_name bbbp"
  "--full_ft --zero_shot --dataset_name toxcast"
)
# 2-2. Random-init으로 full finetuning 하기
TASKS_ckpt_b=(
  "--full_ft --dataset_name hiv --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --dataset_name muv --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --dataset_name sider --batch_size 128 --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --dataset_name clintox --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --dataset_name bace --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --dataset_name tox21 --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --dataset_name bbbp --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --dataset_name toxcast --epoch_freeze 0 --unfreeze_steps 1000"
)

# 3. Gafe Fusion ablation
## 3-1. Full Fine-tuning
### 3-1-1. 2D only
TASKS_ckpt_a+=(
  "--full_ft --fusion_cls 2d_only --dataset_name hiv"
  "--full_ft --fusion_cls 2d_only --dataset_name muv"
  "--full_ft --fusion_cls 2d_only --dataset_name sider --batch_size 64"
  "--full_ft --fusion_cls 2d_only --dataset_name clintox"
  "--full_ft --fusion_cls 2d_only --dataset_name bace"
  "--full_ft --fusion_cls 2d_only --dataset_name tox21"
  "--full_ft --fusion_cls 2d_only --dataset_name bbbp"
  "--full_ft --fusion_cls 2d_only --dataset_name toxcast"
)
### 3-1-1-1. zero shot
TASKS_ckpt_a+=(
  "--full_ft --zero_shot --fusion_cls 2d_only --dataset_name hiv"
  "--full_ft --zero_shot --fusion_cls 2d_only --dataset_name muv"
  "--full_ft --zero_shot --fusion_cls 2d_only --dataset_name sider --batch_size 128"
  "--full_ft --zero_shot --fusion_cls 2d_only --dataset_name clintox"
  "--full_ft --zero_shot --fusion_cls 2d_only --dataset_name bace"
  "--full_ft --zero_shot --fusion_cls 2d_only --dataset_name tox21"
  "--full_ft --zero_shot --fusion_cls 2d_only --dataset_name bbbp"
  "--full_ft --zero_shot --fusion_cls 2d_only --dataset_name toxcast"
)
### 3-1-1-2. random init
TASKS_ckpt_b+=(
  "--full_ft --fusion_cls 2d_only --dataset_name hiv --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 2d_only --dataset_name muv --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 2d_only --dataset_name sider --batch_size 128 --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 2d_only --dataset_name clintox --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 2d_only --dataset_name bace --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 2d_only --dataset_name tox21 --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 2d_only --dataset_name bbbp --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 2d_only --dataset_name toxcast --epoch_freeze 0 --unfreeze_steps 1000"
)
### 3-1-2. 3D only
TASKS_ckpt_a+=(
  "--full_ft --fusion_cls 3d_only --dataset_name hiv"
  "--full_ft --fusion_cls 3d_only --dataset_name muv"
  "--full_ft --fusion_cls 3d_only --dataset_name sider --batch_size 128"
  "--full_ft --fusion_cls 3d_only --dataset_name clintox"
  "--full_ft --fusion_cls 3d_only --dataset_name bace"
  "--full_ft --fusion_cls 3d_only --dataset_name tox21"
  "--full_ft --fusion_cls 3d_only --dataset_name bbbp"
  "--full_ft --fusion_cls 3d_only --dataset_name toxcast"
)
### 3-1-2-1. zero shot
TASKS_ckpt_a+=(
  "--full_ft --zero_shot --fusion_cls 3d_only --dataset_name hiv"
  "--full_ft --zero_shot --fusion_cls 3d_only --dataset_name muv"
  "--full_ft --zero_shot --fusion_cls 3d_only --dataset_name sider --batch_size 128"
  "--full_ft --zero_shot --fusion_cls 3d_only --dataset_name clintox"
  "--full_ft --zero_shot --fusion_cls 3d_only --dataset_name bace"
  "--full_ft --zero_shot --fusion_cls 3d_only --dataset_name tox21"
  "--full_ft --zero_shot --fusion_cls 3d_only --dataset_name bbbp"
  "--full_ft --zero_shot --fusion_cls 3d_only --dataset_name toxcast"
)
### 3-1-2-2. random init
TASKS_ckpt_b+=(
  "--full_ft --fusion_cls 3d_only --dataset_name hiv --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 3d_only --dataset_name muv --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 3d_only --dataset_name sider --batch_size 128 --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 3d_only --dataset_name clintox --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 3d_only --dataset_name bace --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 3d_only --dataset_name tox21 --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 3d_only --dataset_name bbbp --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls 3d_only --dataset_name toxcast --epoch_freeze 0 --unfreeze_steps 1000"
)
### 3-1-3. simple MLP
TASKS_ckpt_a+=(
  "--full_ft --fusion_cls simple_mlp --dataset_name hiv"
  "--full_ft --fusion_cls simple_mlp --dataset_name muv"
  "--full_ft --fusion_cls simple_mlp --dataset_name sider --batch_size 64"
  "--full_ft --fusion_cls simple_mlp --dataset_name clintox"
  "--full_ft --fusion_cls simple_mlp --dataset_name bace"
  "--full_ft --fusion_cls simple_mlp --dataset_name tox21"
  "--full_ft --fusion_cls simple_mlp --dataset_name bbbp"
  "--full_ft --fusion_cls simple_mlp --dataset_name toxcast"
)
### 3-1-3-1. zero shot
TASKS_ckpt_a+=(
  "--full_ft --zero_shot --fusion_cls simple_mlp --dataset_name hiv"
  "--full_ft --zero_shot --fusion_cls simple_mlp --dataset_name muv"
  "--full_ft --zero_shot --fusion_cls simple_mlp --dataset_name sider --batch_size 128"
  "--full_ft --zero_shot --fusion_cls simple_mlp --dataset_name clintox"
  "--full_ft --zero_shot --fusion_cls simple_mlp --dataset_name bace"
  "--full_ft --zero_shot --fusion_cls simple_mlp --dataset_name tox21"
  "--full_ft --zero_shot --fusion_cls simple_mlp --dataset_name bbbp"
  "--full_ft --zero_shot --fusion_cls simple_mlp --dataset_name toxcast"
)
### 3-1-3-2. random init
TASKS_ckpt_b+=(
  "--full_ft --fusion_cls simple_mlp --dataset_name hiv --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls simple_mlp --dataset_name muv --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls simple_mlp --dataset_name sider --batch_size 128 --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls simple_mlp --dataset_name clintox --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls simple_mlp --dataset_name bace --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls simple_mlp --dataset_name tox21 --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls simple_mlp --dataset_name bbbp --epoch_freeze 0 --unfreeze_steps 1000"
  "--full_ft --fusion_cls simple_mlp --dataset_name toxcast --epoch_freeze 0 --unfreeze_steps 1000"
)
## 3-2. Linear Probing
### 3-2-1. 2D only
TASKS_ckpt_a+=(
  "--fusion_cls 2d_only --dataset_name hiv"
  "--fusion_cls 2d_only --dataset_name muv"
  "--fusion_cls 2d_only --dataset_name sider --batch_size 128"
  "--fusion_cls 2d_only --dataset_name clintox"
  "--fusion_cls 2d_only --dataset_name bace"
  "--fusion_cls 2d_only --dataset_name tox21"
  "--fusion_cls 2d_only --dataset_name bbbp"
  "--fusion_cls 2d_only --dataset_name toxcast"
)
### 3-2-2. 3D only
TASKS_ckpt_a+=(
  "--fusion_cls 3d_only --dataset_name hiv"
  "--fusion_cls 3d_only --dataset_name muv"
  "--fusion_cls 3d_only --dataset_name sider --batch_size 128"
  "--fusion_cls 3d_only --dataset_name clintox"
  "--fusion_cls 3d_only --dataset_name bace"
  "--fusion_cls 3d_only --dataset_name tox21"
  "--fusion_cls 3d_only --dataset_name bbbp"
  "--fusion_cls 3d_only --dataset_name toxcast"
)
### 3-2-3. simple MLP
TASKS_ckpt_a+=(
  "--fusion_cls simple_mlp --dataset_name hiv"
  "--fusion_cls simple_mlp --dataset_name muv"
  "--fusion_cls simple_mlp --dataset_name sider --batch_size 128"
  "--fusion_cls simple_mlp --dataset_name clintox"
  "--fusion_cls simple_mlp --dataset_name bace"
  "--fusion_cls simple_mlp --dataset_name tox21"
  "--fusion_cls simple_mlp --dataset_name bbbp"
  "--fusion_cls simple_mlp --dataset_name toxcast"
)

declare -A TASKS_CKPT_MAP
TASKS_CKPT_MAP["$CKPT_A_PATH"]="TASKS_ckpt_a"
TASKS_CKPT_MAP["$CKPT_B_PATH"]="TASKS_ckpt_b"
CKPT_LIST=("$CKPT_A_PATH" "$CKPT_B_PATH")

# ========= 함수들 =========

log() { echo "$@" | tee -a "$RUN_LOG"; }

ckpt_dir_from_path() {
    local ckpt_path="$1"
    basename "$ckpt_path" | sed 's/[^a-zA-Z0-9]/_/g'
}

log_header() {
    local run_time="$1"
    echo "Run at        : $run_time"
    echo "CKPTs         :"
    for ckpt in "${CKPT_LIST[@]}"; do
        echo "  $ckpt"
        echo "${TASKS_CKPT_MAP[$ckpt]}" | sed 's/^/    /'
    done
    echo "Log directory : $LOG_DIR"
    echo ""
}

log_file_from_args() {
    local args_str="$1"
    local ckpt_path="$2"
    local dataset=$(dataset_from_args "$args_str")
    local gs_part=$(gs_print_from_args "$args_str" | sed 's/--//g' | sed 's/ /_/g')
    local prefix="${gs_part:+${gs_part}_}"

    local ckpt_dir=$(ckpt_dir_from_path "$ckpt_path")
    local log_subdir="${LOG_DIR}/${ckpt_dir}"
    mkdir -p "$log_subdir"

    echo "${log_subdir}/${prefix}${dataset}.log"
}

json_file_from_args() {
    local args_str="$1"
    local dataset=$(dataset_from_args "$args_str")
    local gs_part=$(gs_print_from_args "$args_str" | sed 's/--//g' | sed 's/ /_/g')
    local prefix="${gs_part:+${gs_part}_}"

    echo "${JSON_DIR}/${prefix}${dataset}.json"
}


dataset_from_args() {
    local args_str="$1"
    echo "$args_str" | awk '{for(i=1;i<=NF;i++) if($i=="--dataset_name") print $(i+1)}'
}

gs_print_from_args() {
    local args_str="$1"
    local result=""

    for key in "${GS_KEYS[@]}"; do
        if echo "$args_str" | grep -q -- "--$key"; then
            result+="--$key"
            local val=$(echo "$args_str" | awk -v k="--$key" '{for(i=1;i<=NF;i++) if($i==k) print $(i+1)}')
            if [ -n "$val" ] && [[ "$val" != "--"* ]]; then
                result+=" $val"
            fi
            result+=" "
        fi
    done

    echo "$result" | sed 's/  */ /g' | sed 's/ *$//'
}


run_task_on_gpu() {
    local GPU_ID=$1
    local CKPT=$2
    local ARGS_STR="$3"
    local PORT=$4

    local DATASET=$(dataset_from_args "$ARGS_STR")
    local GS_PRINT=$(gs_print_from_args "$ARGS_STR")
    local LOG_FILE=$(log_file_from_args "$ARGS_STR" "$CKPT")
    local SHORT_LOG_FILE="${LOG_FILE##*/}"
    local JSON_PATH=$(json_file_from_args "$ARGS_STR")

    printf -v DATASET_PAD "%-7s" "$DATASET"
    log "[gpu ${GPU_ID}] Dataset: ${DATASET_PAD} | GS: ${GS_PRINT} | CKPT: ${CKPT##*/} | Port: ${PORT} | LOG: ${SHORT_LOG_FILE}"
    local CMD="CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=1 --master_port=$PORT ft_lightning.py --save_metrics_json \"$JSON_PATH\" --pretrain_ckpt \"$CKPT\" $ARGS_STR > \"$LOG_FILE\" 2>&1 &"

    if [ "$TEST_MODE" = true ]; then
        echo "$CMD"
        echo "# (TEST_MODE) Command previewd but not executed:" >> "$LOG_FILE"
        echo "$CMD" >> "$LOG_FILE"
        echo "# (TEST_MODE) JSON path:" >> "$LOG_FILE"
        echo "$JSON_PATH" >> "$LOG_FILE"
        echo ""
        echo ""
        GPU_PIDS[$GPU_ID]=""  # 즉시 완료 처리
    else
        eval "${CMD%&}" &
        GPU_PIDS[$GPU_ID]=$!
    fi

    GPU_TASKS[$GPU_ID]="$CKPT|$ARGS_STR"
    GPU_PORTS[$GPU_ID]=$PORT
}

# ========= TASK 목록 만들기 =========

declare -a ALL_TASKS  # Each item: "ckpt|args"

for CKPT in "${CKPT_LIST[@]}"; do
    TASK_ARRAY_NAME="${TASKS_CKPT_MAP[$CKPT]}"
    eval "TASK_ARRAY=(\"\${${TASK_ARRAY_NAME}[@]}\")"
    for LINE in "${TASK_ARRAY[@]}"; do
        [[ -z "$LINE" ]] && continue
        ALL_TASKS+=( "$CKPT|$LINE" )
    done
done

NUM_TASKS=${#ALL_TASKS[@]}

# ========= 로그 헤더 =========

RUN_LOG="${LOG_DIR}/run_log.log"
log_header "$(date '+%Y-%m-%d %H:%M:%S')" > "$RUN_LOG"

# ========= 초기 실행 =========

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    if [ $NEXT_TASK_INDEX -lt $NUM_TASKS ]; then
        IFS="|" read -r CKPT ARGS <<< "${ALL_TASKS[$NEXT_TASK_INDEX]}"
        PORT=$((BASE_PORT + NEXT_TASK_INDEX))
        run_task_on_gpu $GPU_ID "$CKPT" "$ARGS" $PORT
        ((NEXT_TASK_INDEX++))
        sleep 1
    fi
done

# ========= 메인 루프 =========

while :; do
    ALL_DONE=true
    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        PID=${GPU_PIDS[$GPU_ID]}

        # TEST_MODE인 경우: PID 무시하고 바로 끝난 것으로 처리
        if [ "$TEST_MODE" = true ]; then
            GPU_PIDS[$GPU_ID]=""
            GPU_TASKS[$GPU_ID]=""
            GPU_PORTS[$GPU_ID]=""

            if [ $NEXT_TASK_INDEX -lt $NUM_TASKS ]; then
                IFS="|" read -r CKPT ARGS <<< "${ALL_TASKS[$NEXT_TASK_INDEX]}"
                PORT=$((BASE_PORT + NEXT_TASK_INDEX))
                run_task_on_gpu $GPU_ID "$CKPT" "$ARGS" $PORT
                ((NEXT_TASK_INDEX++))
                ALL_DONE=false
            fi

        elif [ -n "$PID" ]; then
            if ! kill -0 $PID 2>/dev/null; then
                wait $PID
                EXIT_CODE=$?
                IFS="|" read -r CKPT ARGS <<< "${GPU_TASKS[$GPU_ID]}"
                PORT=${GPU_PORTS[$GPU_ID]}
                DATASET=$(dataset_from_args "$ARGS")
                GS_PRINT=$(gs_print_from_args "$ARGS")
                LOG_FILE=$(log_file_from_args "$ARGS" "$CKPT")

                printf -v DATASET_PAD "%-7s" "$DATASET"

                if [ $EXIT_CODE -eq 0 ]; then
                    FLAG="finished successfully ✅"
                else
                    FLAG="failed with exit code $EXIT_CODE ❌"
                    echo "[FAILED] task '${DATASET}' | GS: ${GS_PRINT} | Port: ${PORT} | $FLAG" >> "$LOG_FILE"
                fi

                log "[gpu ${GPU_ID}] Dataset: ${DATASET_PAD} | GS: ${GS_PRINT} | CKPT: ${CKPT##*/} | Port: ${PORT} | $FLAG"

                if [ $NEXT_TASK_INDEX -lt $NUM_TASKS ]; then
                    IFS="|" read -r CKPT ARGS <<< "${ALL_TASKS[$NEXT_TASK_INDEX]}"
                    PORT=$((BASE_PORT + NEXT_TASK_INDEX))
                    run_task_on_gpu $GPU_ID "$CKPT" "$ARGS" $PORT
                    ((NEXT_TASK_INDEX++))
                else
                    GPU_PIDS[$GPU_ID]=""
                    GPU_TASKS[$GPU_ID]=""
                    GPU_PORTS[$GPU_ID]=""
                fi
            else
                ALL_DONE=false
            fi
        fi
    done

    if $ALL_DONE && [ $NEXT_TASK_INDEX -ge $NUM_TASKS ]; then
        break
    fi
    sleep 1
done


log "🎉 All tasks finished."
