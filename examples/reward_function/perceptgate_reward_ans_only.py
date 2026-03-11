# PerceptGate Ablation: curriculum + R_ans only (no R_len, no R_perc)
# LEN_WEIGHT=0, PERC_WEIGHT=0

import re
from typing import Any

import numpy as np

from mathruler.grader import extract_boxed_content, grade_answer

K_ROLLOUTS = 8
FORMAT_WEIGHT = 0.1


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

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        acc = accuracy_reward(response, reward_input["ground_truth"])
        fmt = format_reward(response)
        mar = reward_input.get("mar", None)

        overall = acc + format_weight * fmt

        scores.append({
            "overall": float(overall),
            "accuracy": float(acc),
            "format": float(fmt),
            "r_len": 0.0,
            "r_perc": 0.0,
            "mar": float(mar) if mar is not None else 0.0,
            "mar_available": float(mar is not None),
            "difficulty": 0.0,
        })

    return scores
