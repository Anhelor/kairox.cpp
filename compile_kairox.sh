#!/bin/bash

if ! dpkg -s libssl-dev >/dev/null 2>&1; then
    apt update >/dev/null 2>&1
    apt install -y libssl-dev >/dev/null 2>&1
fi

release_dir=build_rel
debug_dir=build

cmake_opts=(
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_CUDA=ON
    -DGGML_CUDA_GRAPHS=OFF
    -DCMAKE_CUDA_ARCHITECTURES=native
)

usage() {
    echo "usage: $0 <release(rel)|debug(dbg)>"
    exit 1
}

mode=${1-}
[[ $# -eq 1 ]] || usage

case "$mode" in
release | rel)
    build_dir="$release_dir"
    build_type=Release
    ;;
debug | dbg)
    build_dir="$debug_dir"
    build_type=Debug
    cmake_opts+=(-DGGML_CUDA_DEBUG=ON)
    ;;
*)
    usage
    ;;
esac

cmake -B "$build_dir" -DCMAKE_BUILD_TYPE="$build_type" "${cmake_opts[@]}"
cmake --build "$build_dir" --config "$build_type" -j"$(nproc)" \
    --target llama-completion llama-speculative llama-quantize
