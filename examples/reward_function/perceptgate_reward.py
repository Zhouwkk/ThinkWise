# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PerceptGate Reward Function (MAR-based, 方案B)

奖励结构：
    R_total = R_ans + R_len + R_perc

感知信号：MAR（Modality Attention Ratio），由 compute_log_probs 阶段提取后传入。
若 MAR 不可用（extract_mar=false），退化为仅使用 R_ans + format。

四象限 R_len（含难度调制 d_i = 1 - PassRate）：
    状态A（MAR高 + 答对）：+clip(1 - ρ, 0, 1)                      奖励简洁
    状态B（MAR高 + 答错）：+d_i * clip(ρ - 1, 0, 1)                难度加权探索
    状态C（MAR低 + 答对）：-(1+d_i)/2 * clip(ρ - 1, 0, 1)          难度加权惩罚
    状态D（MAR低 + 答错）：0                                         零梯度

R_perc（感知奖惩）：
    MAR > τ_mar：+γ1 * MAR
    MAR ≤ τ_mar：-γ2 * (τ_mar - MAR)
"""

import re
from typing import Any

import numpy as np

from mathruler.grader import extract_boxed_content, grade_answer


# ─── 超参数 ────────────────────────────────────────────────────────────────────
K_ROLLOUTS = 8       # 每道题的 rollout 数
ALPHA = 1.0          # 自适应阈值系数：τ_mar = μ_mar - α * σ_mar
GAMMA1 = 1.0         # R_perc 正向激励系数（MAR 已归一化到 [0,1]）
GAMMA2 = 1.0         # R_perc 负向惩罚系数（MAR 已归一化到 [0,1]）
FORMAT_WEIGHT = 0.1  # 格式奖励权重
LEN_WEIGHT = 0.1     # 长度奖励权重
PERC_WEIGHT = 0.1    # 感知奖励权重
# ──────────────────────────────────────────────────────────────────────────────


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = FORMAT_WEIGHT) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for PerceptGate reward function.")

    num_responses = len(reward_inputs)
    if num_responses == 0:
        return []

    # ── Step 1: 逐条提取基础特征 ──────────────────────────────────────────────
    processed = []
    current_correct = 0
    current_count = 0
    problem_pass_rates = []
    problem_mar_lists = []   # 每道题的 MAR 列表（用于计算自适应阈值）
    current_mar_list = []

    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        length = len(response)
        acc = accuracy_reward(response, reward_input["ground_truth"])
        fmt = format_reward(response)
        mar = reward_input.get("mar", None)  # None 表示 MAR 不可用

        processed.append({"acc": acc, "fmt": fmt, "len": length, "mar": mar})

        if acc == 1.0:
            current_correct += 1
        if mar is not None:
            current_mar_list.append(mar)

        current_count += 1
        if current_count == K_ROLLOUTS:
            problem_pass_rates.append(current_correct / K_ROLLOUTS)
            problem_mar_lists.append(current_mar_list[:])
            current_correct = 0
            current_count = 0
            current_mar_list = []

    # 处理最后一组不足 K_ROLLOUTS 的情况
    if current_count > 0:
        problem_pass_rates.append(current_correct / current_count)
        problem_mar_lists.append(current_mar_list[:])

    # ── Step 2: 计算每道题的自适应 MAR 阈值和难度系数 ────────────────────────
    mar_available = any(d["mar"] is not None for d in processed)

    problem_tau_mar = []   # 每道题的自适应阈值 τ_mar
    problem_difficulty = []  # d_i = 1 - PassRate

    for i, pass_rate in enumerate(problem_pass_rates):
        d_i = 1.0 - pass_rate
        problem_difficulty.append(d_i)

        mar_list = problem_mar_lists[i]
        if mar_list:
            mu = np.mean(mar_list)
            sigma = np.std(mar_list)
            tau = mu - ALPHA * sigma
        else:
            tau = None  # MAR 不可用
        problem_tau_mar.append(tau)

    # ── MAR 归一化到 [0, 1] ──
    if problem_mar_lists and any(problem_mar_lists):
        all_mars = [m for mar_list in problem_mar_lists for m in mar_list]
        mar_min = min(all_mars)
        mar_max = max(all_mars)
        mar_range = mar_max - mar_min if mar_max > mar_min else 1.0
    else:
        mar_min = 0.0
        mar_range = 1.0

    # ── 每道题内部的平均长度（group 级别，与 τ_mar、d_i 对齐） ──
    num_problems = len(problem_pass_rates)
    problem_avg_len = []
    for i in range(num_problems):
        start = i * K_ROLLOUTS
        end = min(start + K_ROLLOUTS, num_responses)
        lens = [processed[j]["len"] for j in range(start, end)]
        problem_avg_len.append(np.mean(lens) if lens else 1.0)

    # ── Step 3: 计算每条轨迹的奖励 ───────────────────────────────────────────
    scores = []

    for idx, data in enumerate(processed):
        problem_idx = idx // K_ROLLOUTS
        acc = data["acc"]
        fmt = data["fmt"]
        length = data["len"]
        mar = data["mar"]
        d_i = problem_difficulty[problem_idx] if problem_idx < len(problem_difficulty) else 0.5
        tau_mar = problem_tau_mar[problem_idx] if problem_idx < len(problem_tau_mar) else None

        rho = length / problem_avg_len[problem_idx] if problem_avg_len[problem_idx] > 0 else 1.0

        # ── R_ans ──
        r_ans = acc

        # ── R_len（四象限，含难度调制）──
        r_len = 0.0
        mar_high = (mar is not None and tau_mar is not None and mar > tau_mar)
        mar_low = (mar is not None and tau_mar is not None and mar <= tau_mar)

        if mar_high and acc == 1.0:
            # 状态A：感知有效 + 答对 → 奖励简洁，惩罚冗长
            r_len = np.clip(1.0 - rho, -1.0, 1.0)
        elif mar_high and acc == 0.0:
            # 状态B：感知有效 + 答错 → 难度加权探索
            r_len = d_i * np.clip(rho - 1.0, 0.0, 1.0)
        elif mar_low and acc == 1.0:
            # 状态C：感知无效 + 答对 → 难度加权惩罚，系数在(-1,0]
            r_len = -((1.0 + d_i) / 2.0) * np.clip(rho - 1.0, 0.0, 1.0)
        else:
            # 状态D：感知无效 + 答错，或 MAR 不可用 → 零梯度
            r_len = 0.0

        # ── R_perc（感知奖惩，仅答对时给正分）──
        r_perc = 0.0
        if mar is not None and tau_mar is not None:
            mar_normalized = (mar - mar_min) / mar_range  # [0, 1]
            if mar > tau_mar and acc == 1.0:
                # 答对 + 视觉依赖达标：正向激励
                r_perc = GAMMA1 * mar_normalized
            elif mar <= tau_mar and acc == 1.0:
                # 答对 + 视觉忽略：惩罚（对冲状态C的虚假正确性）
                r_perc = -GAMMA2 * ((tau_mar - mar_min) / mar_range)
            # 答错时 r_perc = 0，不给无差别加分

        # ── 总奖励 ──
        overall = r_ans + format_weight * fmt + LEN_WEIGHT * r_len + PERC_WEIGHT * r_perc

        scores.append({
            "overall": float(overall),
            "accuracy": float(acc),
            "format": float(fmt),
            "r_len": float(r_len),
            "r_perc": float(r_perc),
            "mar": float(mar) if mar is not None else 0.0,
            "mar_available": float(mar_available),
            "difficulty": float(d_i),
        })

    return scores
