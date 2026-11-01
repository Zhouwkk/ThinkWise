#!/bin/bash
# Naive REINFORCE++ Baseline (7B): R_ans + R_fmt only, no MAR, no curriculum

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
GPU_IDS=(${CUDA_VISIBLE_DEVICES//,/ })
N_GPUS="${N_GPUS:-${#GPU_IDS[@]}}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning:PIL}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"

RUN_ROOT="${RUN_ROOT:-./runs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_ROOT}/checkpoints/thinkwise-7b/reinforce-pp-baseline-7b}"
mkdir -p "${RUN_ROOT}/hf_cache" "${RUN_ROOT}/pip_cache" "${RUN_ROOT}/tensorboard" "${RUN_ROOT}/wandb" "${RUN_ROOT}/logs" "${CHECKPOINT_DIR}"

export HF_HOME="${HF_HOME:-${RUN_ROOT}/hf_cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${RUN_ROOT}/pip_cache}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-${RUN_ROOT}/tensorboard}"
export WANDB_DIR="${WANDB_DIR:-${RUN_ROOT}/wandb}"

if [ "${FRESH_START:-0}" = "1" ]; then
    RESUME_ARGS=(
        "trainer.find_last_checkpoint=false"
        "trainer.load_checkpoint_path=null"
    )
else
    if [ -f "${CHECKPOINT_DIR}/checkpoint_tracker.json" ]; then
        RESUME_ARGS=( "trainer.find_last_checkpoint=true" )
    else
        RESUME_ARGS=(
            "trainer.find_last_checkpoint=false"
            "trainer.load_checkpoint_path=null"
        )
    fi
fi

python -m verl.trainer.main \
    config=examples/config_thinkwise.yaml \
    "trainer.n_gpus_per_node=${N_GPUS}" \
    "trainer.save_checkpoint_path=${CHECKPOINT_DIR}" \
    trainer.project_name=thinkwise-mar \
    trainer.experiment_name=reinforce-pp-baseline-7b \
    trainer.save_limit=-1 \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.online_filtering=false \
    algorithm.hperc_filter_schedule=null \
    "worker.actor.model.model_path=${MODEL_PATH}" \
    worker.reward.reward_function=./examples/reward_function/grpo_baseline_reward.py:compute_score \
    "${RESUME_ARGS[@]}" \
    "$@" 2>&1 | tee "${RUN_ROOT}/logs/train_reinforce_pp_baseline_7b.log"
