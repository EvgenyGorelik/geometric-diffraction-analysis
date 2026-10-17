#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: ./run.sh [--local] [--mount HOST:CONTAINER[:ro]] [--gpus auto|none|all|DEVICE] COMMAND [ARGS...]

Commands (remaining arguments are forwarded unchanged):
  simulate              Configured rotation experiment (default config if omitted)
  simulate-random       Original random-orientation training-data simulator
  train                 Classification training
  train-segmentation    Segmentation training
  infer                 Classification inference
  infer-segmentation    Segmentation inference
  split                 Create classification train/validation split
  prepare-segmentation  Prepare segmentation data
  convert-tif           Convert TIFF images to PNG
  synthetic             Convert experimental images to synthetic style
  stats                 Compute image statistics
  overlay               Create image overlays
  extract-peaks         Extract diffraction peaks
  test                  Run regression tests
  python                Run an arbitrary Python command/module
  shell                 Open bash inside the container
  build                 Build/update the Docker image; extra args go to docker build
  help                  Show this help

By default, build the image if missing, then run in Docker. The repository is
mounted at /workspace, outputs persist, and Linux files use your UID/GID.
Relative paths refer to the repository root. Mount outside data paths explicitly:
  ./run.sh --mount /data:/data simulate --hkl-file /data/output.hkl --intensity-mode hkl
  ./run.sh train --root_dir data/processed --num_epochs 8
  ./run.sh --gpus all train --root_dir data/processed
  ./run.sh build --build-arg TORCH_BACKEND=cpu
  ./run.sh --local simulate --start-deg 0 --stop-deg 0 --output /tmp/example

Environment: GDA_IMAGE (default geometric-diffraction:latest), GDA_GPUS (auto),
GDA_SHM_SIZE (2g), PYTHON (python3, local/container interpreter).
GPU auto mode enables all GPUs for training/inference only when nvidia-smi works.
Explicit GPU use requires NVIDIA Container Toolkit. --local uses installed Python.
Documentation: docs/README.md
HELP
}

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
local_mode=0
mounts=()
gpus="${GDA_GPUS:-auto}"
while (($#)); do
    case "$1" in
        --local) local_mode=1; shift ;;
        --mount|--gpus)
            option="$1"
            if (($# < 2)); then echo "Missing value for $option" >&2; exit 2; fi
            if [[ "$option" == --mount ]]; then mounts+=(--volume "$2"); else gpus="$2"; fi
            shift 2 ;;
        *) break ;;
    esac
done
command="${1:-help}"
if (($#)); then shift; fi
case "$command" in
    help|-h|--help) usage; exit 0 ;;
    simulate|simulate-random|train|train-segmentation|infer|infer-segmentation|split|prepare-segmentation|convert-tif|synthetic|stats|overlay|extract-peaks|test|python|shell|build) ;;
    *) echo "Unknown command: $command. Run ./run.sh help." >&2; exit 2 ;;
esac

if [[ "${GDA_IN_CONTAINER:-0}" == 1 || "$local_mode" == 1 ]]; then
    if [[ "$local_mode" == 1 && ${#mounts[@]} -gt 0 ]]; then
        echo '--mount is only supported in Docker mode' >&2; exit 2
    fi
    if [[ "${GDA_IN_CONTAINER:-0}" == 1 ]]; then repo_dir="${GDA_PROJECT_DIR:-/workspace}"; fi
    cd -- "$repo_dir"
    export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
    export MPLBACKEND="${MPLBACKEND:-Agg}"
    python_bin="${PYTHON:-python3}"
    case "$command" in
        simulate)
            # Use the lightweight entry point; legacy generation has its own command.
            has_config=0
            for arg in "$@"; do
                case "$arg" in --config|--config=*) has_config=1 ;; esac
            done
            if [[ "$has_config" == 0 ]]; then set -- --config configs/rotation_experiment.json "$@"; fi
            exec "$python_bin" -m simulation.rotation_experiment "$@" ;;
        simulate-random) module=simulation.simulate_experiment ;;
        train) module=training.train ;;
        train-segmentation) module=training.train_segmentation ;;
        infer) module=inference.run_inference ;;
        infer-segmentation) module=inference.infer_segmentation ;;
        split) module=data_processing.create_data_split ;;
        prepare-segmentation) module=utils.prepare_segmentation_data ;;
        convert-tif) module=data_processing.convert_tif ;;
        synthetic) module=data_processing.create_synthetic_images ;;
        stats) module=data_processing.get_img_stats ;;
        overlay) module=data_processing.create_overlay ;;
        extract-peaks) module=utils.extract_peaks ;;
        test) exec "$python_bin" -m unittest discover -s tests -v "$@" ;;
        python) exec "$python_bin" "$@" ;;
        shell) exec bash "$@" ;;
        build) echo 'Run build on the host without --local.' >&2; exit 2 ;;
    esac
    exec "$python_bin" -m "$module" "$@"
fi

if ! command -v docker >/dev/null 2>&1; then
    echo 'Docker is required. Install Docker or use ./run.sh --local COMMAND with dependencies installed.' >&2
    exit 127
fi
image="${GDA_IMAGE:-geometric-diffraction:latest}"
if [[ "$command" == build ]]; then
    exec docker build -f "$repo_dir/docker/Dockerfile" -t "$image" "$@" "$repo_dir"
fi
if ! docker image inspect "$image" >/dev/null 2>&1; then
    docker build -f "$repo_dir/docker/Dockerfile" -t "$image" "$repo_dir"
fi
run_args=(run --rm --init --shm-size "${GDA_SHM_SIZE:-2g}")
# Avoid root-owned experiment and checkpoint files on Linux bind mounts.
if [[ "$(uname -s)" == Linux ]]; then run_args+=(--user "$(id -u):$(id -g)"); fi
if [[ -t 0 ]]; then run_args+=(-i); fi
if [[ -t 0 && -t 1 ]]; then run_args+=(-t); fi
if [[ "$gpus" == auto ]]; then
    gpus=none
    case "$command" in
        train|train-segmentation|infer|infer-segmentation)
            if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then gpus=all; fi ;;
    esac
fi
if [[ "$gpus" != none ]]; then run_args+=(--gpus "$gpus"); fi
run_args+=(--volume "$repo_dir:/workspace" --workdir /workspace)
run_args+=("${mounts[@]}")
exec docker "${run_args[@]}" "$image" "$command" "$@"
