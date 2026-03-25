"""
Standard GRPO Baseline Reward Function

R_total = R_ans + 0.1 * R_fmt

纯正确性奖励 + 格式奖励，无长度奖励、无感知奖励。
作为 baseline 与 PerceptGate 各消融实验对比。
"""

import re
from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer

K_ROLLOUTS = 8
FORMAT_WEIGHT = 0.1


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
        raise ValueError("Please use `reward_type=batch` for this reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        acc = accuracy_reward(response, reward_input["ground_truth"])
        fmt = format_reward(response)
        length = len(response)

        overall = acc + format_weight * fmt
        
        # Add a tiny global length penalty to prevent natural verbosity drift in PPO
        overall -= 0.01 * (length / 4096.0)

        scores.append({
            "overall": float(overall),
            "accuracy": float(acc),
            "format": float(fmt),
            "r_len": 0.0,
            "r_perc": 0.0,
            "mar": 0.0,
            "mar_available": 0.0,
            "difficulty": 0.0,
        })

    return scores
