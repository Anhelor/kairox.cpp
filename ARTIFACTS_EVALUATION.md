# KAIROX Artifact Evaluation Guide

This README describes the KAIROX artifact repository for OSDI'26 artifact evaluation. KAIROX is an adaptive GPU-CPU hybrid LLM inference system. The artifact is intended to validate the released implementation and check selected Figure 9 throughput trends for KAIROX against the included comparison systems (`llama_cpp` and `neuralink`).

No separate clone of upstream llama.cpp is required; the artifact repository contains the modified llama.cpp tree, the KAIROX implementation, and the artifact-evaluation scripts.

## Badge Target and Scope

This artifact is submitted only for the OSDI'26 Artifacts Available badge. The artifact repository is:

```text
https://github.com/Anhelor/kairox.cpp
```

The repository will remain publicly readable throughout the artifact-evaluation period.

The required artifact-evaluation path covers:

- `simple_validation`: a short execution check that builds the artifact and runs `llama_cpp`, `kairox`, and `neuralink` on one validation model for the selected hardware profile: Prosparse-LLaMA2-7B for `low` and Prosparse-LLaMA2-13B for `high`.
- `e2e_performance`: end-to-end completion and speculative-decoding throughput runs for KAIROX and the included comparison systems (`llama_cpp` and `neuralink`), corresponding to selected Figure 9 trends in the paper.

The required software path is self-contained in this repository, except for external model weights and standard container/dependency downloads described below. The paper also compares against the open-source [PowerInfer](https://github.com/Tiiny-AI/PowerInfer) and [Q-Infer](https://github.com/PDS-Lab/Q-Infer) systems. Those external systems are discussed in the paper, but evaluators do not need to clone, rebuild, or configure them for the required artifact-evaluation path.

The accepted KAIROX paper PDF is submitted through HotCRP for OSDI artifact review and is not stored in this GitHub repository. The artifact does not intentionally perform destructive operations, collect analytics, or track evaluators. It writes only build outputs, benchmark logs, result summaries, and downloaded model files in the paths described here.

## Repository Contents

The repository contains:

- the modified llama.cpp source tree with the KAIROX implementation.
- `ARTIFACTS_EVALUATION.md`: this artifact evaluation guide.
- `bench_models.sh`: the top-level validation and benchmark driver.
- `test_kairox.sh`: the per-backend runner used by `bench_models.sh`.
- `compile_kairox.sh`: the build helper used by the evaluation scripts.
- `prompts.txt`: the fixed prompt set used by benchmark mode.
- `parse_bench_logs.py`: a helper that parses benchmark logs and emits Markdown and CSV summaries.

Model weights are not stored in this GitHub repository. Download them from the model repository described below.

## Hardware, Software, and Time Requirements

The paper evaluates two consumer-grade PC profiles:

- `low`: PC-Low, RTX 3080 Ti, 12 GB VRAM, 12 benchmark CPU threads plus at least one additional host thread, PCIe 3.0 x16.
- `high`: PC-High, RTX 4090, 24 GB VRAM, 16 benchmark CPU threads plus at least one additional host thread, PCIe 4.0 x16.

For the closest hardware match, run on RTX 3080 Ti and RTX 4090 systems. Different CPUs, host memory bandwidth, PCIe generation, drivers, or model subsets may change absolute throughput numbers.

Additional requirements:

- Docker and NVIDIA Container Toolkit.
- An NVIDIA driver compatible with CUDA 12.8.
- Enough disk space for the source tree, build outputs, logs, and the selected GGUF model files.

The quick `simple_validation` path is intended to complete within a short time after dependencies and models are available. The full `e2e_performance` path depends on the selected profile, hardware, model subset, and whether both profiles are run.

## Setup

On the host, clone the artifact, prepare the model directory, and start the CUDA container:

```sh
git clone https://github.com/Anhelor/kairox.cpp.git
cd kairox.cpp

docker pull pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel

mkdir -p "$HOME/SPIF-GGUF"

docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -it --rm \
    -v "$PWD":/workspace/kairox.cpp \
    -v "$HOME/SPIF-GGUF":/root/SPIF-GGUF \
    pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel \
    bash
```

If the cloned artifact directory is not the current host directory, replace `"$PWD"` with its absolute path.

Inside the container, enter the artifact directory, install dependencies, and confirm that the GPU is visible:

```sh
cd /workspace/kairox.cpp

apt update
apt install -y git git-lfs cmake build-essential libssl-dev
python -m pip install -U "huggingface_hub[hf_xet]" --break-system-packages
git lfs install

nvidia-smi
```

Download the SPIF GGUF artifacts to the path expected by `bench_models.sh`. For the full `e2e_performance` path, download all active model files:

```sh
hf download Anhelor/SPIF-GGUF \
    --repo-type model \
    --local-dir /root/SPIF-GGUF
```

All active model entries in `bench_models.sh` must have corresponding case-sensitive files under `/root/SPIF-GGUF`.

For a smaller validation-only run on PC-Low, download only the `low` validation subset:

```sh
hf download Anhelor/SPIF-GGUF \
    --repo-type model \
    --include "prosparse-llama-2-7b.gguf" \
    --include "Llama-160M-Chat-v1-Q8_0.gguf" \
    --include "prosparse-llama-2-7b-sparkinfer-model-split-688.gguf" \
    --local-dir /root/SPIF-GGUF
```

For a smaller validation-only run on PC-High, download only the `high` validation subset:

```sh
hf download Anhelor/SPIF-GGUF \
    --repo-type model \
    --include "prosparse-llama-2-13b.gguf" \
    --include "Llama-160M-Chat-v1-Q8_0.gguf" \
    --include "prosparse-llama-2-13b-sparkinfer-model-split-864.gguf" \
    --local-dir /root/SPIF-GGUF
```

## Getting Started Instructions

Run one of the short validation commands inside the container:

```sh
cd /workspace/kairox.cpp
test -f prompts.txt && echo "prompts.txt found"

# PC-Low / RTX 3080 Ti
bash bench_models.sh low simple

# PC-High / RTX 4090
bash bench_models.sh high simple
```

This builds the release binaries with `compile_kairox.sh rel`, then runs completion and speculative decoding for `llama_cpp`, `kairox`, and `neuralink` on the profile's validation model: Prosparse-LLaMA2-7B for `low` and Prosparse-LLaMA2-13B for `high`. This path checks that the artifact builds and executes on the selected profile; it is not intended to reproduce the paper's throughput conclusions.

The `simple_validation` run passes if the script exits with status code 0, the build succeeds, all configured backends finish without runtime errors, logs are written, and each log ends with a `benchmark summary` block.

## Detailed Instructions

Run one of the full benchmark profiles inside the container:

```sh
# PC-Low / RTX 3080 Ti
bash bench_models.sh low full

# PC-High / RTX 4090
bash bench_models.sh high full
```

The full path runs `simple_validation` followed by `e2e_performance`. The benchmark runner uses a maximum context size of 1024 tokens and a maximum output length of 512 tokens, matching the Figure 9 setup. Logs are written to:

- `low_logs/` for the PC-Low profile.
- `high_logs/` for the PC-High profile.

Each log filename has this schema:

```text
<benchmark_group>__<backend>__<hardware>__<generation_mode>__<model>.log
```

Expected benchmark groups are `simple_validation` and `e2e_performance`. Expected backends are `llama_cpp`, `kairox`, and `neuralink`. Expected generation modes are `completion` and `speculative`; models without a compatible draft model are skipped for speculative decoding.

The `e2e_performance` run passes if the script exits with status code 0, logs are generated for the configured model/backend/generation-mode combinations, `parse_bench_logs.py` can summarize the logs, and KAIROX reports higher `decode mean` than the included comparison systems for matching profile, model, and generation mode:

```text
kairox > llama_cpp AND kairox > neuralink
```

## Result Checking

Benchmark logs end with a `benchmark summary` block. A shortened example is:

```text
benchmark summary:
  target measured runs: 10
  attempted runs:       13
  filtered runs (decode < 16): 3
  included runs:        10
  prefill mean:   42.33 t/s
  decode mean:    10.91 t/s
```

`decode mean` is the main throughput metric used to compare KAIROX with the included comparison systems under the same profile, model, and generation mode.

To inspect summaries directly:

```sh
grep -R -A 8 "benchmark summary:" low_logs/ high_logs/
```

If only one profile was run, pass only that profile's log directory.

To generate Markdown and CSV summaries:

```sh
python parse_bench_logs.py low_logs high_logs --out results_summary.md --csv results_summary.csv
```

The parser reads `.log` files, extracts the last `benchmark summary` block from each log, and reports each row's benchmark group, backend, profile/hardware, generation mode, model, `decode mean`, and speedups relative to matching `llama_cpp` rows when available.

## Troubleshooting and Limitations

- GPU is not visible: check the host NVIDIA driver, Docker, and NVIDIA Container Toolkit, then restart the container with `--gpus all`.
- Hugging Face download is interrupted: rerun the same `hf download` command.
- Disk space is insufficient: check both Docker storage and the filesystem backing `/root/SPIF-GGUF`.
- Model file is reported missing: compare the failing path with:

```sh
ls /root/SPIF-GGUF
```

- Compilation fails: confirm that dependency installation completed and that the GPU is visible inside the container.
- Hardware differs from the paper's PC-Low or PC-High machines: absolute throughput may differ, but the run is still useful for checking relative behavior.
