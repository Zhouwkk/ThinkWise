#!/bin/bash
# 消融实验 E2 (3B): curriculum + R_ans + R_len (no R_perc)

export CUDA_VISIBLE_DEVICES=0,4
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
CHECKPOINT_DIR=${RUN_ROOT}/checkpoints/perceptgate/pg-curriculum-ans-len-3b
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
    trainer.n_gpus_per_node=2 \
    trainer.save_checkpoint_path=${CHECKPOINT_DIR} \
    trainer.project_name=perceptgate-mar \
    trainer.experiment_name=pg-curriculum-ans-len-3b \
    trainer.save_limit=-1 \
    algorithm.online_filtering=true \
    algorithm.hperc_filter_schedule=null \
    worker.reward.reward_function=./examples/reward_function/perceptgate_reward_ans_len.py:compute_score \
    "${RESUME_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train_curriculum_ans_len_3b.log"
