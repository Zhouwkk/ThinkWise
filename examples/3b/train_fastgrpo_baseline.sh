#!/bin/bash

set -x

# 先清理可能残留的 Ray 进程
# ray stop --force 2>/dev/null
# sleep 3

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
GPU_IDS=(${CUDA_VISIBLE_DEVICES//,/ })
N_GPUS="${N_GPUS:-${#GPU_IDS[@]}}"

export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning:PIL}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

RUN_ROOT="${RUN_ROOT:-./runs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints/fast-grpo/fg-step3-3b-baseline}"

mkdir -p "${RUN_ROOT}/hf_cache" "${RUN_ROOT}/pip_cache" "${RUN_ROOT}/tensorboard" "${RUN_ROOT}/wandb" "${RUN_ROOT}/logs" "${CHECKPOINT_DIR}"

export HF_HOME="${HF_HOME:-${RUN_ROOT}/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUN_ROOT}/pip_cache}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_ROOT}/tensorboard}"
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"

if [ "${FRESH_START:-0}" = "1" ]; then
    # Force a fresh run: do not load any previous checkpoint.
    RESUME_ARGS=(
        "trainer.find_last_checkpoint=false"
        "trainer.load_checkpoint_path=null"
    )
else
    if [ -f "${CHECKPOINT_DIR}/checkpoint_tracker.json" ]; then
        RESUME_ARGS=(
            "trainer.find_last_checkpoint=true"
        )
    else
        RESUME_ARGS=(
            "trainer.load_checkpoint_path=null"
            "trainer.find_last_checkpoint=false"
        )
    fi
fi

# Keep data/batch/length settings aligned with ThinkWise config,
# only switch method-specific components to FAST-GRPO baseline.
METHOD_ARGS=(
    "worker.reward.reward_function=./examples/reward_function/thinking_reward.py:compute_score"
    "algorithm.hperc_filter_schedule=null"
    "algorithm.kl_type=group_accuracy_based"
    "trainer.project_name=fast-grpo"
    "trainer.experiment_name=fg-step3-3b-baseline"
)

python -m verl.trainer.main \
    config=examples/config_thinkwise.yaml \
    "trainer.n_gpus_per_node=${N_GPUS}" \
    "trainer.save_checkpoint_path=${CHECKPOINT_DIR}" \
    trainer.save_limit=-1 \
    "${METHOD_ARGS[@]}" \
    "${RESUME_ARGS[@]}" \
    "$@" 2>&1 | tee "${RUN_ROOT}/logs/train_fastgrpo_baseline_2gpu.log"
