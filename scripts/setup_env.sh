#!/bin/bash
# 重建 fast_grpo 环境，锁定关键版本避免冲突
set -e

ENV_NAME=fast_grpo

# 设置缓存目录到 /data，避免 home 空间不足
export TMPDIR=/data/zhouwenkang/tmp
export PIP_CACHE_DIR=/data/zhouwenkang/.cache/pip
mkdir -p $TMPDIR $PIP_CACHE_DIR

# 删除旧环境
conda deactivate 2>/dev/null || true
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# 创建新环境
conda create -n $ENV_NAME python=3.11 -y
conda activate $ENV_NAME

# 1. 先装 torch（锁定版本）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2. 装 flash-attn（从源码编译，基于 torch 2.6.0）
pip install flash-attn --no-build-isolation --no-cache-dir

# 3. 装 vllm（锁定 0.8.5）
pip install vllm==0.8.5

# 4. 装其余依赖
pip install accelerate codetiming datasets liger-kernel mathruler numpy omegaconf \
    pandas peft pillow "pyarrow>=15.0.0" pylatexenc qwen-vl-utils "ray[default]" \
    tensordict torchdata "transformers>=4.54.0,<=4.57.0" wandb scipy

echo "Done! Verify:"
python -c "import torch; print('torch:', torch.__version__)"
python -c "import vllm; print('vllm:', vllm.__version__)"
python -c "import flash_attn; print('flash_attn:', flash_attn.__version__)"
