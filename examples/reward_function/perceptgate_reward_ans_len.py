# PerceptGate Ablation: curriculum + R_ans + R_len (no R_perc)
# LEN_WEIGHT=0.5, PERC_WEIGHT=0
# 与完整版 perceptgate_reward.py 相同，仅关闭 R_perc

import re
from typing import Any

import numpy as np

from mathruler.grader import extract_boxed_content, grade_answer

K_ROLLOUTS = 8
ALPHA = 1.0
FORMAT_WEIGHT = 0.1
LEN_WEIGHT = 0.1
PERC_WEIGHT = 0.0  # 关闭 R_perc


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

    # ── 每道题内部的平均长度（group 级别，与 τ_mar、d_i 对齐） ──
    num_problems = len(problem_pass_rates)
    problem_avg_len = []
    for i in range(num_problems):
        start = i * K_ROLLOUTS
        end = min(start + K_ROLLOUTS, num_responses)
        lens = [processed[j]["len"] for j in range(start, end)]
        problem_avg_len.append(np.mean(lens) if lens else 1.0)

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

        r_ans = acc

        # R_len（四象限）
        r_len = 0.0
        mar_high = (mar is not None and tau_mar is not None and mar > tau_mar)
        mar_low = (mar is not None and tau_mar is not None and mar <= tau_mar)

        if mar_high and acc == 1.0:
            r_len = np.clip(1.0 - rho, -1.0, 1.0)
        elif mar_high and acc == 0.0:
            r_len = d_i * np.clip(rho - 1.0, 0.0, 1.0)
        elif mar_low and acc == 1.0:
            r_len = -((1.0 + d_i) / 2.0) * np.clip(rho - 1.0, 0.0, 1.0)
        else:
            r_len = 0.0

        # R_perc = 0（本消融关闭）
        r_perc = 0.0

        overall = r_ans + format_weight * fmt + LEN_WEIGHT * r_len

        scores.append({
            "overall": float(overall),
            "accuracy": float(acc),
            "format": float(fmt),
            "r_len": float(r_len),
            "r_perc": 0.0,
            "mar": float(mar) if mar is not None else 0.0,
            "mar_available": float(mar_available),
            "difficulty": float(d_i),
        })

    return scores
