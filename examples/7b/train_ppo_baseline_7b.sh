#!/bin/bash
# Naive PPO Baseline (7B): R_ans + R_fmt only, no MAR, no curriculum

# Physical GPUs 1,2,4,5 (processes see them as cuda:0..3)
export CUDA_VISIBLE_DEVICES=1,2,4,5
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
CHECKPOINT_DIR=${RUN_ROOT}/checkpoints/perceptgate-7b/ppo-baseline-7b
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
    trainer.project_name=perceptgate-mar \
    trainer.experiment_name=ppo-baseline-7b \
    trainer.save_limit=-1 \
    algorithm.adv_estimator=gae \
    algorithm.gamma=0.99 \
    algorithm.lam=0.95 \
    algorithm.kl_coef=0.05 \
    algorithm.online_filtering=false \
    algorithm.hperc_filter_schedule=null \
    worker.rollout.n=1 \
    worker.rollout.disable_tqdm=true \
    worker.rollout.gpu_memory_utilization=0.35 \
    worker.actor.global_batch_size=8 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.fsdp.enable_cpu_offload=true \
    worker.actor.offload.offload_params=false \
    worker.actor.offload.offload_optimizer=false \
    worker.actor.mar_mode=disabled \
    worker.actor.model.model_path=/data/zhouwenkang/models/Qwen2.5-VL-7B-Instruct \
    worker.critic.model.model_path=/data/zhouwenkang/models/Qwen2.5-VL-7B-Instruct \
    worker.critic.global_batch_size=8 \
    worker.critic.fsdp.enable_cpu_offload=true \
    worker.critic.micro_batch_size_per_device_for_update=2 \
    worker.critic.micro_batch_size_per_device_for_experience=1 \
    worker.critic.cliprange_value=0.2 \
    worker.critic.ppo_epochs=2 \
    worker.reward.reward_function=./examples/reward_function/grpo_baseline_reward.py:compute_score \
    "${RESUME_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train_ppo_baseline_7b.log"
