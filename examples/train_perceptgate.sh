#!/bin/bash

# set -x

# # 先清理可能残留的 Ray 进程
# ray stop --force 2>/dev/null
# sleep 3

# 使用 GPU 2,6（GPU 3 硬件异常，无法创建 CUDA context）
export CUDA_VISIBLE_DEVICES=2,6

# NCCL 设置：GPU 2,6 跨 NUMA（SYS），无 NVLink，无 InfiniBand
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=ens12f1np1

# Reduce CUDA memory fragmentation from eager attention in compute_mar
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 统一把训练产物写到 /data-store，避免 /data 分区写满
RUN_ROOT=/data-store/zhouwenkang/fast_runs
CHECKPOINT_DIR=${RUN_ROOT}/checkpoints/perceptgate/pg-step2-3b
MIGRATION_CKPT=/data/zhouwenkang/FAST/checkpoints/perceptgate/pg-step2-3b/global_step_20

mkdir -p "${RUN_ROOT}/hf_cache" "${RUN_ROOT}/pip_cache" "${RUN_ROOT}/tensorboard" "${RUN_ROOT}/wandb" "${RUN_ROOT}/logs" "${CHECKPOINT_DIR}"

export HF_HOME=${RUN_ROOT}/hf_cache
export PIP_CACHE_DIR=${RUN_ROOT}/pip_cache
export TENSORBOARD_DIR=${RUN_ROOT}/tensorboard
export WANDB_DIR=${RUN_ROOT}/wandb

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
            "trainer.load_checkpoint_path=${MIGRATION_CKPT}"
            "trainer.find_last_checkpoint=false"
        )
    fi
fi

python -m verl.trainer.main \
    config=examples/config_perceptgate.yaml \
    trainer.n_gpus_per_node=2 \
    trainer.save_checkpoint_path=${CHECKPOINT_DIR} \
    trainer.project_name=perceptgate-mar \
    trainer.experiment_name=pg-len_group-curriculum \
    trainer.save_limit=-1 \
    algorithm.online_filtering=true \
    algorithm.hperc_filter_schedule=null \
    "${RESUME_ARGS[@]}" 2>&1 | tee "${RUN_ROOT}/logs/train_2gpu.log"
