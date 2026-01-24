#!/bin/bash

PROFILE=${1:-low}
MODEL_ROOT=/root/SPIF-GGUF
USAGE="usage: $0 <low|high> [simple|full]"

if [[ $# -gt 2 ]]; then
    echo "$USAGE" >&2
    exit 1
fi

case "$PROFILE" in
low)
    PLATFORM=3080ti
    BENCH_RUNS=10
    LOG_DIR=low_logs
    MODELS=(
        "$MODEL_ROOT/Bamboo-base-v0_1.gguf|$MODEL_ROOT/Lite-Mistral-150M-v2-Instruct-32000-Q8_0.gguf|$MODEL_ROOT/Bamboo-base-v0_1-sparkinfer-model-split-896.gguf"
        "$MODEL_ROOT/SparseQwen2-7B.gguf|$MODEL_ROOT/Qwen2-0.5B-Instruct-Q8_0.gguf|$MODEL_ROOT/SparseQwen2-7B-sparkinfer-model-split-592.gguf"
        "$MODEL_ROOT/prosparse-llama-2-7b.gguf|$MODEL_ROOT/Llama-160M-Chat-v1-Q8_0.gguf|$MODEL_ROOT/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf"
        "$MODEL_ROOT/opt-6.7b.gguf|$MODEL_ROOT/opt-125m-Q8_0.gguf|$MODEL_ROOT/opt-6.7b-sparkinfer-model-split-1024.gguf"
        "$MODEL_ROOT/prosparse-llama-2-13b.gguf|$MODEL_ROOT/Llama-160M-Chat-v1-Q8_0.gguf|$MODEL_ROOT/prosparse-llama-2-13b-sparkinfer-model-split-864.gguf"
        "$MODEL_ROOT/opt-13b.gguf|$MODEL_ROOT/opt-125m-Q8_0.gguf|$MODEL_ROOT/opt-13b-sparkinfer-model-split-1024.gguf"
        "$MODEL_ROOT/opt-30b-Q4_K_M.gguf|$MODEL_ROOT/opt-125m-Q8_0.gguf|$MODEL_ROOT/opt-30b-sparkinfer-model-split-1024.gguf"
    )
    VALIDATION_MODEL="$MODEL_ROOT/prosparse-llama-2-7b.gguf|$MODEL_ROOT/Llama-160M-Chat-v1-Q8_0.gguf|$MODEL_ROOT/prosparse-llama-2-7b-sparkinfer-model-split-688.gguf"
    ;;
high)
    PLATFORM=4090
    BENCH_RUNS=20
    LOG_DIR=high_logs
    MODELS=(
        "$MODEL_ROOT/prosparse-llama-2-13b.gguf|$MODEL_ROOT/Llama-160M-Chat-v1-Q8_0.gguf|$MODEL_ROOT/prosparse-llama-2-13b-sparkinfer-model-split-864.gguf"
        "$MODEL_ROOT/ReluFalcon-40B-Q8_0.gguf||$MODEL_ROOT/ReluFalcon-40B-sparkinfer-model-split-1024.gguf"
        "$MODEL_ROOT/opt-13b.gguf|$MODEL_ROOT/opt-125m-Q8_0.gguf|$MODEL_ROOT/opt-13b-sparkinfer-model-split-1024.gguf"
        "$MODEL_ROOT/opt-30b-Q8_0.gguf|$MODEL_ROOT/opt-125m-Q8_0.gguf|$MODEL_ROOT/opt-30b-sparkinfer-model-split-1024.gguf"
        "$MODEL_ROOT/opt-66b-Q4_K_M.gguf|$MODEL_ROOT/opt-125m-Q8_0.gguf|$MODEL_ROOT/opt-66b-sparkinfer-model-split-1024.gguf"
    )
    VALIDATION_MODEL="$MODEL_ROOT/prosparse-llama-2-13b.gguf|$MODEL_ROOT/Llama-160M-Chat-v1-Q8_0.gguf|$MODEL_ROOT/prosparse-llama-2-13b-sparkinfer-model-split-864.gguf"
    ;;
*)
    echo "$USAGE" >&2
    exit 1
    ;;
esac

RUN_MODE=${2:-full}
RUNNER=./test_kairox.sh
BACKENDS=(llama_cpp kairox neuralink)
KINDS=(completion speculative)

case "$RUN_MODE" in
simple | full) ;;
*)
    echo "$USAGE" >&2
    exit 1
    ;;
esac

mkdir -p "$LOG_DIR"

run_case() {
    local group=$1
    local backend=$2
    local kind=$3
    local model=$4
    local draft_model=$5
    local model_split=$6
    local bench_mode=$7
    local model_name log_name log_path

    local args=(
        "$RUNNER"
        "$backend"
        "$PLATFORM"
        "kind=$kind"
        "vb=0"
        "model=$model"
    )

    if ((bench_mode)); then
        args+=(bench)
    fi

    if [[ "$kind" == "speculative" ]]; then
        args+=("draft_model=$draft_model")
    fi

    if [[ "$backend" != "llama_cpp" ]]; then
        args+=("model_split=$model_split")
    fi

    model_name=$(basename "$model" .gguf)
    log_name="${group}__${backend}__${PLATFORM}__${kind}__${model_name}.log"
    log_path="$LOG_DIR/$log_name"

    if ((bench_mode)); then
        args+=("bench_runs=$BENCH_RUNS")
    fi

    bash "${args[@]}" >"$log_path" 2>&1
}

run_entries_kind_cases() {
    local group=$1
    local bench_mode=$2
    shift 2
    local backend kind entry model draft_model model_split

    for backend in "${BACKENDS[@]}"; do
        for kind in "${KINDS[@]}"; do
            for entry in "$@"; do
                IFS='|' read -r model draft_model model_split <<<"$entry"
                if [[ "$kind" == "speculative" && -z "$draft_model" ]]; then
                    continue
                fi
                run_case "$group" "$backend" "$kind" "$model" "$draft_model" "$model_split" "$bench_mode"
            done
        done
    done
}

run_simple_validation_cases() {
    run_entries_kind_cases simple_validation 0 "$VALIDATION_MODEL"
}

run_e2e_performance_cases() {
    run_entries_kind_cases e2e_performance 1 "${MODELS[@]}"
}

rm -rf build_rel && bash compile_kairox.sh rel
run_simple_validation_cases
if [[ "$RUN_MODE" == "full" ]]; then
    run_e2e_performance_cases
fi
