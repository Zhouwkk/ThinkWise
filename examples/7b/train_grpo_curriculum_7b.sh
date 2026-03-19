#!/bin/bash
# GRPO + Curriculum (7B): R_ans + R_fmt + 课程采样，无MAR

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1
export TQDM_DISABLE=1
export PYTHONWARNINGS="ignore::UserWarning:PIL"
export RAY_DEDUP_LOGS=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=ens12f1np1
export PYTORCH_ALLOC_CONF=expandable_segments:True

RUN_ROOT=/data-store/zhouwenkang/fast_runs
CHECKPOINT_DIR=${RUN_ROOT}/checkpoints/perceptgate-7b/grpo-curriculum-7b
mkdir -p "${RUN_ROOT}/hf_cache" "${RUN_ROOT}/pip_cache" "${RUN_ROOT}/tensorboard" "${RUN_ROOT}/wandb" "${RUN_ROOT}/logs" "${CHECKPOINT_DIR}"

export HF_HOME=${RUN_ROOT}/hf_cache
export PIP_CACHE_DIR=${RUN_ROOT}/pip_cache
export TENSORBOARD_DIR=${RUN_ROOT}/tensorboard
export WANDB_DIR=${RUN_ROOT}/wandb

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
    config=examples/config_perceptgate.yaml \
    trainer.n_gpus_per_node=4 \
    trainer.save_checkpoint_path=${CHECKPOINT_DIR} \
    trainer.project_name=perceptgate-mar-7b \
    trainer.experiment_name=grpo-curriculum-7b \
    trainer.save_limit=-1 \
    algorithm.online_filtering=true \
    algorithm.hperc_filter_schedule=null \
    worker.actor.model.model_path=/data/zhouwenkang/models/Qwen2.5-VL-7B-Instruct \
    worker.reward.reward_function=./examples/reward_function/grpo_baseline_reward.py:compute_score \
    "${RESUME_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train_grpo_curriculum_7b.log"
