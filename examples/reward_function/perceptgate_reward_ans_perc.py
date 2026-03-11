# PerceptGate Ablation: curriculum + R_ans + R_perc (no R_len)
# LEN_WEIGHT=0, PERC_WEIGHT=0.1
# 用于验证 R_len 的贡献

import re
from typing import Any

import numpy as np

from mathruler.grader import extract_boxed_content, grade_answer

K_ROLLOUTS = 8
ALPHA = 1.0
GAMMA1 = 1.0
GAMMA2 = 1.0
FORMAT_WEIGHT = 0.1
LEN_WEIGHT = 0.0    # 关闭 R_len
PERC_WEIGHT = 0.1


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = FORMAT_WEIGHT,
) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for PerceptGate reward function.")

    num_responses = len(reward_inputs)
    if num_responses == 0:
        return []

    processed = []
    current_correct = 0
    current_count = 0
    problem_pass_rates = []
    problem_mar_lists = []
    current_mar_list = []

    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        length = len(response)
        acc = accuracy_reward(response, reward_input["ground_truth"])
        fmt = format_reward(response)
        mar = reward_input.get("mar", None)

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

    if current_count > 0:
        problem_pass_rates.append(current_correct / current_count)
        problem_mar_lists.append(current_mar_list[:])

    mar_available = any(d["mar"] is not None for d in processed)

    problem_tau_mar = []
    problem_difficulty = []

    for i, pass_rate in enumerate(problem_pass_rates):
        d_i = 1.0 - pass_rate
        problem_difficulty.append(d_i)

        mar_list = problem_mar_lists[i]
        if mar_list:
            mu = np.mean(mar_list)
            sigma = np.std(mar_list)
            tau = mu - ALPHA * sigma
        else:
            tau = None
        problem_tau_mar.append(tau)

    # MAR 归一化到 [0, 1]
    if problem_mar_lists and any(problem_mar_lists):
        all_mars = [m for mar_list in problem_mar_lists for m in mar_list]
        mar_min = min(all_mars)
        mar_max = max(all_mars)
        mar_range = mar_max - mar_min if mar_max > mar_min else 1.0
    else:
        mar_min = 0.0
        mar_range = 1.0

    scores = []

    for idx, data in enumerate(processed):
        problem_idx = idx // K_ROLLOUTS
        acc = data["acc"]
        fmt = data["fmt"]
        mar = data["mar"]
        d_i = problem_difficulty[problem_idx] if problem_idx < len(problem_difficulty) else 0.5
        tau_mar = problem_tau_mar[problem_idx] if problem_idx < len(problem_tau_mar) else None

        r_ans = acc

        # R_perc（感知奖惩，仅答对时给正分）
        r_perc = 0.0
        if mar is not None and tau_mar is not None:
            mar_normalized = (mar - mar_min) / mar_range
            if mar > tau_mar and acc == 1.0:
                r_perc = GAMMA1 * mar_normalized
            elif mar <= tau_mar and acc == 1.0:
                r_perc = -GAMMA2 * ((tau_mar - mar_min) / mar_range)

        overall = r_ans + format_weight * fmt + PERC_WEIGHT * r_perc

        scores.append({
            "overall": float(overall),
            "accuracy": float(acc),
            "format": float(fmt),
            "r_len": 0.0,
            "r_perc": float(r_perc),
            "mar": float(mar) if mar is not None else 0.0,
            "mar_available": float(mar_available),
            "difficulty": float(d_i),
        })

    return scores
