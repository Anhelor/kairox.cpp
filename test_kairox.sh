#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

seed=42
ctx_size=1024
max_tokens=512
bench_runs=20
prompt='Implement and compare multiple sorting algorithms in Python, including quicksort, mergesort, heapsort, and insertion sort. For each algorithm, provide clean implementations, analyze time and space complexity, and discuss when it performs best.

```python'

usage() {
    cat <<'EOF'
usage:
  test_kairox.sh <backend> <platform> kind=<completion|speculative> vb=<GiB> [bench] [model=PATH] [draft_model=PATH] [model_split=PATH] [bench_runs=N] [kairox_dfr_lambda_*=...]

platforms:
  3080ti -> gpu_vram=12, threads=12
  4090   -> gpu_vram=24, threads=16

backends:
  llama_cpp   llama.cpp path, uses -fit on; vb=0 maps to -fitt 512
  kairox  KAIROX path, requires model_split
  neuralink   KAIROX path, requires model_split

profile differences:
  backend     lambda adapt
  kairox      0.67   0.05
  neuralink   0.00   0.00
EOF
    exit 1
}

require_arg() {
    [[ -n "$1" ]] || usage
}

require_numeric() {
    case "$1" in
    '' | *[!0-9]*)
        usage
        ;;
    esac
}

validate_args() {
    case "$backend" in
    llama_cpp | kairox | neuralink) ;;
    *)
        usage
        ;;
    esac

    case "$kind" in
    completion | speculative) ;;
    *)
        usage
        ;;
    esac

    require_arg "$model"
    require_arg "$vb"
    require_numeric "$vb"

    [[ "$kind" != "speculative" || -n "$draft_model" ]] || usage

    case "$backend" in
    llama_cpp)
        if [[ "$vb" -gt "$gpu_vram" ]]; then
            echo "error: vb=$vb exceeds gpu_vram=$gpu_vram for platform=$platform" >&2
            exit 1
        fi
        ;;
    kairox | neuralink)
        require_arg "$model_split"
        ;;
    esac
}

set_platform_defaults() {
    case "$platform" in
    3080ti)
        gpu_vram=12
        threads=12
        ;;
    4090)
        gpu_vram=24
        threads=16
        ;;
    *)
        usage
        ;;
    esac
}

set_backend_defaults() {
    kairox_parallel=0
    kairox_dfr_lambda_init=0.67
    kairox_dfr_lambda_adapt_rate=0.05

    case "$backend" in
    llama_cpp) ;;
    kairox)
        kairox_parallel=1
        ;;
    neuralink)
        kairox_parallel=1
        kairox_dfr_lambda_init=0.00
        kairox_dfr_lambda_adapt_rate=0.00
        ;;
    *)
        usage
        ;;
    esac
}

parse_args() {
    backend=${1-}
    platform=${2-}
    bench_flag=0
    kind=
    vb=
    draft_model=
    model=
    model_split=

    set_platform_defaults
    set_backend_defaults

    extra_args=("${@:3}")
    for arg in "${extra_args[@]}"; do
        case "$arg" in
        bench)
            bench_flag=1
            ;;
        kind=*)
            kind=${arg#kind=}
            ;;
        vb=*)
            vb=${arg#vb=}
            ;;
        draft_model=*)
            draft_model=${arg#draft_model=}
            ;;
        model=*)
            model=${arg#model=}
            ;;
        model_split=*)
            model_split=${arg#model_split=}
            ;;
        bench_runs=*)
            bench_runs=${arg#bench_runs=}
            ;;
        kairox_dfr_lambda_init=*)
            kairox_dfr_lambda_init=${arg#*=}
            ;;
        kairox_dfr_lambda_adapt_rate=*)
            kairox_dfr_lambda_adapt_rate=${arg#*=}
            ;;
        *)
            usage
            ;;
        esac
    done

    validate_args
}

build_env_args() {
    env_args=("CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")

    if [[ "$backend" != "llama_cpp" ]]; then
        env_args+=(
            "KAIROX_PARALLEL=$kairox_parallel"
            "KAIROX_DFR_LAMBDA_INIT=$kairox_dfr_lambda_init"
            "KAIROX_DFR_LAMBDA_ADAPT_RATE=$kairox_dfr_lambda_adapt_rate"
        )
    fi
}

build_cmd_args() {
    bin_dir="./build_rel/bin"

    shared_opts=(
        -t "$threads"
        -s "$seed"
        -p "$prompt"
        -c "$ctx_size"
        -n "$max_tokens"
        --no-warmup
    )

    if [[ "$backend" == "llama_cpp" ]]; then
        backend_opts=(
            -fit on
            -ngl auto
        )
        if ((vb == 0)); then
            backend_opts+=(-fitt 512)
        else
            fit_target_mib=$(((gpu_vram - vb) * 1024))
            backend_opts+=(-fitt "$fit_target_mib")
        fi
    else
        backend_opts=(
            -kairox-ms "$model_split"
            -cffn -fit off -ngl all
            --no-mmap --no-direct-io
            -vb "$vb"
        )
    fi

    case "$kind" in
    completion)
        cmd_args=(
            "$bin_dir/llama-completion"
            -m "$model"
            -no-cnv
            --repeat-penalty 1.1
            --dry-multiplier 0.6
            "${backend_opts[@]}"
            "${shared_opts[@]}"
        )
        ;;
    speculative)
        cmd_args=(
            "$bin_dir/llama-speculative"
            -md "$draft_model"
            -ngld all
            -m "$model"
            --draft-min 3
            --draft-max 5
            --repeat-penalty 1.15
            --dry-multiplier 0.8
            "${backend_opts[@]}"
            "${shared_opts[@]}"
        )
        ;;
    *)
        usage
        ;;
    esac

    if ((bench_flag)); then
        cmd_args+=(
            --bench-prompt-file prompts.txt
            --bench-warmup 2
            --bench-runs "$bench_runs"
            --bench-no-print
        )
    fi
}

print_cmd() {
    printf '%q ' env "${env_args[@]}" "${cmd_args[@]}"
    printf '\n'
}

run_cmd() {
    env "${env_args[@]}" "${cmd_args[@]}"
}

parse_args "$@"
build_env_args
build_cmd_args
print_cmd
run_cmd
