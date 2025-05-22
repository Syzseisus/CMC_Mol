#!/bin/bash

# ========= 함수들 =========

log_header() {
    local run_time="$1"
    local pretrain_ckpt="$2"
    echo "Run at        : $run_time"
    echo "pretrain_ckpt : $pretrain_ckpt"
    echo "Grid Search Space:"
    for key in "${GRID_SEARCH_KEYS[@]}"; do
        echo "  $key: ${GRID_SEARCH_POOL[$key]}"
    done
    echo ""
}

log_file_from_args() {
    local args_str="$1"
    echo "${LOG_DIR}/$(echo "$args_str" \
        | sed 's/--dataset_name //' \
        | sed 's/--//g' \
        | sed 's/  */_/g' \
        | sed 's/^_//' \
        | sed 's/_$//' ).log"
}

dataset_from_args() {
    local args_str="$1"
    echo "$args_str" | awk '{for(i=1;i<=NF;i++) if($i=="--dataset_name") print $(i+1)}'
}

gs_print_from_args() {
    local args_str="$1"
    # 마지막 key가 dataset_name임을 보장
    local last_key="${GRID_SEARCH_KEYS[-1]}"
    local result=""
    for key in "${GRID_SEARCH_KEYS[@]:0:${#GRID_SEARCH_KEYS[@]}-1}"; do
        local val=$(echo "$args_str" | awk -v k="--$key" '{for(i=1;i<=NF;i++) if($i==k) print $(i+1)}')
        if [ -n "$val" ]; then
            result+="--$key $val "
        fi
    done
    echo "${result::-1}"  # 마지막 공백 제거
}

make_grid_search_tasks() {
    local -n TASKS_REF=$1
    shift
    local keys=("$@")
    TASKS_REF=()
    if [ ${#keys[@]} -eq 0 ]; then
        TASKS_REF+=("")
        return
    fi
    local key=${keys[0]}
    local vals=(${GRID_SEARCH_POOL[$key]})
    local rest_keys=("${keys[@]:1}")
    local sub_name="sub_tasks_$RANDOM"
    local -n sub_tasks=$sub_name
    sub_tasks=()
    make_grid_search_tasks $sub_name "${rest_keys[@]}"
    for v in "${vals[@]}"; do
        for sub in "${sub_tasks[@]}"; do
            if [ -z "$sub" ]; then
                TASKS_REF+=("--$key $v")
            else
                TASKS_REF+=("--$key $v $sub")
            fi
        done
    done
}

############################################
# ========= GRID SEARCH SPACE 정의 =========
############################################

declare -A GRID_SEARCH_POOL
GRID_SEARCH_POOL[split_strat]="force_scaffold"
GRID_SEARCH_POOL[fusion_cls]="2d_only 3d_only simple_mlp"
GRID_SEARCH_POOL[dataset_name]="hiv muv sider clintox bace tox21 bbbp toxcast"

GRID_SEARCH_KEYS=(split_strat fusion_cls)  # 앞을 고정시키고 돌림
GRID_SEARCH_KEYS+=(dataset_name)   # dataset_name은 반드시 마지막!

############################################
############################################

# ========= 기본 세팅 =========

PRETRAIN_CKPT=$1
BASE_LOG_DIR="./run_gs_logs"
TIMESTAMP=$(TZ='Asia/Seoul' date "+%Y%m%d_%H%M%S")
LOG_DIR="${BASE_LOG_DIR}_${TIMESTAMP}"

NUM_GPUS=4
BASE_PORT=29100

mkdir -p "$LOG_DIR"

declare -a GPU_PIDS
declare -a GPU_TASKS
NEXT_TASK_INDEX=0

# ========= TASKS 배열 만들기 =========

TASKS=()
make_grid_search_tasks TASKS "${GRID_SEARCH_KEYS[@]}"
NUM_TASKS=${#TASKS[@]}

# ========= 로그 파일 헤더 =========

RUN_LOG="${LOG_DIR}/run_log.log"
log_header "$(date '+%Y-%m-%d %H:%M:%S')" "$PRETRAIN_CKPT" > "$RUN_LOG"
log() { echo "$@" | tee -a "$RUN_LOG"; }

# ========= GPU에 태우는 함수 =========

run_task_on_gpu() {
    local GPU_ID=$1
    local ARGS_STR="$2"
    local PORT=$3

    local DATASET=$(dataset_from_args "$ARGS_STR")
    local GS_PRINT=$(gs_print_from_args "$ARGS_STR")
    local LOG_FILE=$(log_file_from_args "$ARGS_STR")
    local SHORT_LOG_FILE="${LOG_FILE##*/}"

    printf -v DATASET_PAD "%-7s" "$DATASET"

    log "[gpu ${GPU_ID}] Dataset: ${DATASET_PAD} | GS: ${GS_PRINT} | Port: ${PORT} | LOGGING: ${SHORT_LOG_FILE}"

    # pretrain 뺐음
    CUDA_VISIBLE_DEVICES=$GPU_ID torchrun --nproc_per_node=1 --master_port=$PORT \
        ft_lightning.py \
        --pretrain_ckpt "$PRETRAIN_CKPT" \
        $ARGS_STR \
        > "$LOG_FILE" 2>&1 &

    local PID=$!
    GPU_PIDS[$GPU_ID]=$PID
    GPU_TASKS[$GPU_ID]="$ARGS_STR"
    GPU_PORTS[$GPU_ID]=$PORT
}

# ========= 초기 실행 =========

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    if [ $NEXT_TASK_INDEX -lt $NUM_TASKS ]; then
        ARGS_STR="${TASKS[$NEXT_TASK_INDEX]}"
        PORT=$((BASE_PORT + NEXT_TASK_INDEX))
        run_task_on_gpu $GPU_ID "$ARGS_STR" $PORT
        ((NEXT_TASK_INDEX++))
    fi
done

# ========= 메인 루프 =========

while :; do
    ALL_DONE=true
    for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
        PID=${GPU_PIDS[$GPU_ID]}
        if [ -n "$PID" ]; then
            if ! kill -0 $PID 2>/dev/null; then
                wait $PID
                EXIT_CODE=$?
                ARGS_STR="${GPU_TASKS[$GPU_ID]}"
                PORT=${GPU_PORTS[$GPU_ID]}
                DATASET=$(dataset_from_args "$ARGS_STR")
                GS_PRINT=$(gs_print_from_args "$ARGS_STR")
                LOG_FILE=$(log_file_from_args "$ARGS_STR")

                printf -v DATASET_PAD "%-7s" "$DATASET"

                if [ $EXIT_CODE -eq 0 ]; then
                    FLAG="finished successfully ✅"
                else
                    FLAG="failed with exit code $EXIT_CODE ❌"
                    echo "[FAILED] task '${DATASET}' | GS: ${GS_PRINT} | Port: ${PORT} | $FLAG" >> "$LOG_FILE"
                fi

                log "[gpu ${GPU_ID}] Dataset: ${DATASET_PAD} | GS: ${GS_PRINT} | Port: ${PORT} | $FLAG"

                if [ $NEXT_TASK_INDEX -lt $NUM_TASKS ]; then
                    ARGS_STR="${TASKS[$NEXT_TASK_INDEX]}"
                    PORT=$((BASE_PORT + NEXT_TASK_INDEX))
                    run_task_on_gpu $GPU_ID "$ARGS_STR" $PORT
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
