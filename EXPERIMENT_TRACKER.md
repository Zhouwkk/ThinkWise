# PerceptGate 实验计划与跟踪

## 1. 环境配置

```bash
git clone https://github.com/Zhouwkk/PerceptGate.git
cd PerceptGate

conda create -n perceptgate python=3.11
conda activate perceptgate

pip install -r requirements.txt
pip install -e .
```

---

## 2. 数据准备

训练和验证数据集不包含在本仓库中，需自行下载并转换为兼容格式。

### 2.1 数据格式要求

训练代码通过 `verl/utils/dataset.py` 中的 `RLHFDataset` 加载数据，支持 JSON / JSONL / Parquet 格式。每条样本需包含以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | str | 问题文本，多模态样本需包含 `<image>` 占位符标记图片插入位置 |
| `answer` | str | 标准答案，需用 `\boxed{}` 包裹（如 `\boxed{42}`） |
| `image` | list[str] | 图片路径列表（相对于 `image_dir` 的相对路径，或绝对路径） |

示例：
```json
{
  "question": "<image>\nWhat is the value of x in the triangle shown above?",
  "answer": "\\boxed{60}",
  "image": ["images/triangle_001.png"]
}
```

对于纯文本样本，省略 `image` 字段或传空列表即可。`question` 中的 `<image>` 数量需与 `image` 列表长度一致。

训练时会自动拼接 format prompt（`examples/format_prompt/math.jinja`），在问题末尾追加 `Please reason step by step, and put your final answer within \boxed{}`。

### 2.2 训练数据：ViRL39K

来源：[TIGER-Lab/ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K)

```bash
# 1. 下载 parquet 文件到本地
#    从 HuggingFace 下载 39Krelease.parquet

# 2. 转换为训练格式
python scripts/prepare_virl39k.py \
    --input /path/to/39Krelease.parquet \
    --output /path/to/virl39k_train.json
```

该脚本会：
- 为缺少 `<image>` 占位符的 question 自动补全
- 将 parquet 中的图片数据转为路径列表
- 输出 JSON 格式供 `RLHFDataset` 加载

### 2.3 验证数据：MMK12

来源：[FanqingM/MMK12](https://huggingface.co/datasets/FanqingM/MMK12)

```bash
python scripts/prepare_mmk12_val.py \
    --output_dir /path/to/MMK12 \
    --hf_token <your_token>  # 可选
```

该脚本会：
- 从 HuggingFace 下载 MMK12 test split
- 将图片保存到 `output_dir/images/`
- 拼接选项到 question 文本
- 用 `\boxed{}` 包裹答案
- 输出 `mmk12_test.json`

### 2.4 配置数据路径

在 `examples/config_perceptgate.yaml` 中修改：

```yaml
data:
  train_files: /path/to/virl39k_train.json
  val_files:
    - /path/to/MMK12/mmk12_test.json
  image_dir: /path/to/ViRL39K  # 训练集图片根目录
```

### 2.5 路径修改说明

克隆项目后需要修改以下两处绝对路径：

**`examples/config_perceptgate.yaml`（必改）**

```yaml
data:
  train_files: /path/to/virl39k_train.json
  val_files:
    - /path/to/MMK12/mmk12_test.json
  image_dir: /path/to/ViRL39K

worker:
  actor:
    model:
      model_path: /path/to/Qwen2.5-VL-3B-Instruct  # 或 7B-Instruct
```

**所有训练脚本顶部（`examples/*.sh` 和 `examples/7b/*.sh`）**

```bash
RUN_ROOT=/path/to/your/output_dir   # checkpoint、日志、wandb 等产物的输出根目录
```

7B 脚本中还需修改：

```bash
worker.actor.model.model_path=/path/to/Qwen2.5-VL-7B-Instruct
```

### 2.6 自定义数据集适配

如需使用其他数据集，只需确保转换为上述 JSON 格式即可。关键要点：

1. `question` 中每个 `<image>` 对应 `image` 列表中的一张图片
2. `answer` 必须用 `\boxed{}` 包裹，奖励函数通过 `extract_boxed_content()` 提取答案进行评分
3. 图片路径可以是绝对路径（直接使用），也可以是相对路径（会与 `config.data.image_dir` 拼接）
4. 可选字段 `category`、`source` 等会被保留但不影响训练

---

## 2. 统一训练设置

| 项目 | 配置 |
|------|------|
| 基础模型 | Qwen2.5-VL-3B-Instruct / Qwen2.5-VL-7B-Instruct |
| 训练数据 | ViRL39K |
| 验证数据 | MMK12 |
| GPU | A100 80GB × N（按需配置） |
| Rollout K | 8 |
| Batch Size | 512 (rollout) / 8 (global update) |
| 学习率 | 1e-6 |
| Max Prompt Length | 4096 |
| Max Response Length | 4096 |
| 格式奖励权重 | 0.1 |
| MAR 层选取 | Mid（中间50%层） |
| MAR α | 1.0 |
| γ1, γ2 | 1.0 |
| LEN_WEIGHT | 0.1 |
| PERC_WEIGHT | 0.1 |
| 课程采样 schedule | Phase1(0→50%): low=0.01, high=0.7; Phase2(50→100%): low=0.3, high=0.99 |

---

## 3. 实验计划总表

### 3B 模型实验

| ID | 实验名 | 模型 | 奖励组成 | 课程采样 | KL 类型 | 训练脚本 | 状态 | val acc | val len (mean) | 备注 |
|----|--------|------|---------|---------|---------|---------|------|---------|----------------|------|
| 3B-E4 | PerceptGate Full | 3B | R_ans + R_fmt + R_len + R_perc | ✅ | fixed (0.01) | `train_curriculum_full.sh` | 🔲 待运行 | — | — | |

### 7B 模型实验

运行顺序：第一批 E4 + E0，结果出来后再启动 E2 + E3。

| ID | 实验名 | 模型 | 奖励组成 | 课程采样 | KL 类型 | 训练脚本 | 状态 | val acc | val len (mean) | 备注 |
|----|--------|------|---------|---------|---------|---------|------|---------|----------------|------|
| E4 | PerceptGate Full | 7B | R_ans + R_fmt + R_len + R_perc | ✅ | fixed (0.01) | `7b/train_curriculum_full_7b.sh` | 🔲 待运行 | — | — | 第一批：完整版PerceptGate |
| E0 | GRPO Baseline | 7B | R_ans + R_fmt | ❌ | fixed (0.01) | `7b/train_grpo_baseline_7b.sh` | 🔲 待运行 | — | — | 第一批：标准GRPO基线 |
| E2 | Curriculum + R_ans + R_len | 7B | R_ans + R_fmt + R_len | ✅ | fixed (0.01) | `7b/train_curriculum_ans_len_7b.sh` | 🔲 待运行 | — | — | 第二批：消融R_perc |
| E3 | Curriculum + R_ans + R_perc | 7B | R_ans + R_fmt + R_perc | ✅ | fixed (0.01) | `7b/train_curriculum_ans_perc_7b.sh` | 🔲 待运行 | — | — | 第二批：消融R_len |

> 状态标记：🔲 待运行 · 🔄 运行中 · ✅ 已完成 · ❌ 训练终止

---

## 4. 消融对比逻辑

```
E0 (GRPO Baseline)   ← 纯正确性奖励基线
E4 (Full)            ← 完整版 PerceptGate（E4 vs E0）
E2 (- R_perc)        ← 去掉 R_perc，验证其贡献（E4 vs E2）
E3 (- R_len)         ← 去掉 R_len，验证其贡献（E4 vs E3）
```

核心对比：
- **PerceptGate vs Baseline**：E4 vs E0
- **R_perc 贡献**：E4 vs E2
- **R_len 贡献**：E4 vs E3

---

## 5. 待解决问题

1. **数据筛选**：对 ViRL39K 筛选，去除仅通过文本就能做对的样本

---

## 6. 敏感性分析计划

在主实验（E0/E4）完成并验证方法有效后，针对 γ1、γ2 进行敏感性分析。前期 3B 实验观察到 R_perc 对准确率影响更显著，值得单独验证。

| 实验 | γ_len | γ_perc | 目的 |
|------|-------|--------|------|
| S0 | 0.1 | 0.1 | 基准（E4 默认） |
| S1 | 0.1 | 0.2 | R_perc 增强，验证对准确率的影响 |
| S2 | 0.2 | 0.2 | 同步增强，验证长度控制是否改善 |

> 前提：E4 主实验结果出来后，根据 val acc 和 val len 的分化情况决定是否启动。

---

## 7. 训练脚本使用说明

所有训练脚本位于 `examples/` 目录，统一通过环境变量和命令行参数控制行为。

### 7.1 从头开始训练

**3B 模型（PerceptGate Full）：**
```bash
FRESH_START=1 bash examples/train_curriculum_full.sh
```

**7B 模型（PerceptGate Full）：**
```bash
FRESH_START=1 bash examples/7b/train_curriculum_full_7b.sh
```

`FRESH_START=1` 会强制忽略已有 checkpoint，从基础模型开始训练。不设置该变量时，脚本会自动检测 `CHECKPOINT_DIR` 下是否存在 `checkpoint_tracker.json`，有则自动续跑。

### 7.2 从最新 checkpoint 续跑

**3B 模型：**
```bash
bash examples/train_curriculum_full.sh
```

**7B 模型：**
```bash
bash examples/7b/train_curriculum_full_7b.sh
```

脚本默认行为：若 `${CHECKPOINT_DIR}/checkpoint_tracker.json` 存在，自动设置 `trainer.find_last_checkpoint=true`，从最新保存的 step 继续训练。

### 7.3 从指定 checkpoint 迁移

若要从另一个实验的 checkpoint 迁移（如用 step1 的权重初始化 step2），在脚本中修改 `MIGRATION_CKPT` 变量，并确保 `CHECKPOINT_DIR` 下不存在 `checkpoint_tracker.json`：

```bash
# 脚本内部逻辑（无需手动操作，了解即可）
MIGRATION_CKPT=/path/to/other_experiment/global_step_XX
# 当 CHECKPOINT_DIR 下无 checkpoint_tracker.json 且 FRESH_START!=1 时自动使用
```

### 7.4 关键参数说明

**GPU 配置**

```bash
# 修改脚本顶部的 CUDA_VISIBLE_DEVICES，指定使用哪些卡
export CUDA_VISIBLE_DEVICES=0,1,2,3   # 使用 4 卡

# 同步修改 trainer.n_gpus_per_node，与卡数保持一致
trainer.n_gpus_per_node=4
```

**模型路径**

在 `examples/config_perceptgate.yaml` 中修改：

```yaml
worker:
  actor:
    model:
      model_path: /path/to/Qwen2.5-VL-3B-Instruct   # 或 7B-Instruct
```

也可以在命令行覆盖：

```bash
python -m verl.trainer.main \
    config=examples/config_perceptgate.yaml \
    worker.actor.model.model_path=/path/to/Qwen2.5-VL-7B-Instruct \
    ...
```

**序列长度**

```yaml
data:
  max_prompt_length: 4096
  max_response_length: 4096
```

**Batch Size**

```yaml
data:
  rollout_batch_size: 512        # rollout 阶段每步处理的样本总数
  mini_rollout_batch_size: 8     # 单次 rollout forward 的 batch 大小（显存相关）

worker:
  actor:
    global_batch_size: 8         # 梯度更新的全局 batch size
    micro_batch_size_per_device_for_update: 1      # 每卡更新时的 micro batch
    micro_batch_size_per_device_for_experience: 1  # 每卡 rollout 时的 micro batch
```

**wandb 实验命名**

```bash
trainer.project_name=perceptgate-mar    # wandb project
trainer.experiment_name=my-exp-name    # wandb run 名称，同时决定 checkpoint 目录命名
```

**奖励函数**

```bash
worker.reward.reward_function=./examples/reward_function/perceptgate_reward.py:compute_score
```

**课程采样开关**

```bash
algorithm.online_filtering=true    # 开启课程采样
algorithm.online_filtering=false   # 关闭（GRPO baseline 用）
```

### 7.5 日志查看

训练日志同时写入文件、TensorBoard 和 wandb：

```bash
# 实时查看训练日志
tail -f ${RUN_ROOT}/logs/train_curriculum_full.log

# TensorBoard
tensorboard --logdir ${RUN_ROOT}/tensorboard
```

---

## 8. 评测说明

使用 [PAPO-Eval](https://github.com/xhguo7/PAPO-Eval) 进行多数据集评测，具体操作参见该项目文档。

计划评测数据集：

| 数据集 | PAPO-Eval 原生支持 |
|--------|:-----------------:|
| MathVista | ✅ |
| MathVerse | ✅ |
| WeMath | ✅ |
| MMK12 | ✅ |
| MathVision | ❌ 需适配 |
| MM-Math | ❌ 需适配 |
| DynaMath | ❌ 需适配 |
| MM-Vet | ❌ 需适配 |
| Bingo | ❌ 需适配 |
| MMHALU | ❌ 需适配 |
| MMVP | ❌ 需适配 |
| MMEval-Pro | ❌ 需适配 |
