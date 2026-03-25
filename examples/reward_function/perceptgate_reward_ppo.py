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
PerceptGate Reward Function -- PPO version (Scheme A: batch-level global statistics)

Differences from the GRPO version:
  - PPO samples one trajectory per question (rollout.n=1), so group-level statistics
    are unavailable.
  - d_i (difficulty) is estimated from the batch-wide average pass rate: d = 1 - batch_acc
  - rho (relative length) uses the batch-wide average length as the baseline.
  - All other reward logic (four-quadrant R_len, R_perc) is identical to the GRPO version.

Reward structure:
    R_total = R_ans + R_len + R_perc
"""

import re
from typing import Any

import numpy as np

from mathruler.grader import extract_boxed_content, grade_answer


# --- Hyperparameters ----------------------------------------------------------
ALPHA = 1.0          # adaptive threshold coefficient: tau_mar = mu_mar - alpha * sigma_mar
GAMMA1 = 1.0         # R_perc positive incentive coefficient
GAMMA2 = 1.0         # R_perc negative penalty coefficient
FORMAT_WEIGHT = 0.1  # format reward weight
LEN_WEIGHT = 0.1     # length reward weight
PERC_WEIGHT = 0.1    # perception reward weight
REFERENCE_LEN = 1024 # reference length for PPO to break the batch-level feedback loop
# ------------------------------------------------------------------------------


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    return 1.0 if re.fullmatch(pattern, response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = FORMAT_WEIGHT,
    len_weight: float = LEN_WEIGHT,
    perc_weight: float = PERC_WEIGHT,
) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for PerceptGate PPO reward function.")

    num_responses = len(reward_inputs)
    if num_responses == 0:
        return []

    # -- Step 1: extract per-sample features -----------------------------------
    processed = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        length = len(response)
        acc = accuracy_reward(response, reward_input["ground_truth"])
        fmt = format_reward(response)
        mar = reward_input.get("mar", None)
        processed.append({"acc": acc, "fmt": fmt, "len": length, "mar": mar})

    # -- Step 2: batch-level global statistics (replaces GRPO group stats) -----
    # difficulty: d = 1 - batch_acc
    batch_acc = np.mean([d["acc"] for d in processed])
    d_global = 1.0 - batch_acc

    # length baseline: use a fixed reference length for PPO to avoid positive feedback loops
    # batch_avg_len = float(np.mean([d["len"] for d in processed]))
    # batch_avg_len = max(batch_avg_len, 1.0)
    ref_len = float(REFERENCE_LEN)

    # adaptive MAR threshold from all available MAR values in the batch
    mar_values = [d["mar"] for d in processed if d["mar"] is not None]
    mar_available = len(mar_values) > 0

    if mar_available:
        mu_mar = float(np.mean(mar_values))
        sigma_mar = float(np.std(mar_values))
        tau_mar = mu_mar - ALPHA * sigma_mar
        mar_min = float(min(mar_values))
        mar_max = float(max(mar_values))
        mar_range = max(mar_max - mar_min, 1e-8)
    else:
        tau_mar = None
        mar_min = 0.0
        mar_range = 1.0

    # -- Step 3: compute per-sample rewards ------------------------------------
    scores = []
    for data in processed:
        acc = data["acc"]
        fmt = data["fmt"]
        length = data["len"]
        mar = data["mar"]

        rho = length / ref_len
        d_i = d_global  # Scheme A: all samples share the batch-level difficulty estimate

        # R_ans
        r_ans = acc

        # R_len (four-quadrant with difficulty modulation)
        r_len = 0.0
        mar_high = (mar is not None and tau_mar is not None and mar > tau_mar)
        mar_low  = (mar is not None and tau_mar is not None and mar <= tau_mar)

        if mar_high and acc == 1.0:
            # State A: perception valid + correct -> reward conciseness
            r_len = float(np.clip(1.0 - rho, -1.0, 1.0))
        elif mar_high and acc == 0.0:
            # State B: perception valid + wrong -> difficulty-weighted exploration
            # Cap the exploration reward to prevent length explosion in PPO (n=1)
            r_len = d_i * float(np.clip(rho - 1.0, 0.0, 0.5))
        elif mar_low and acc == 1.0:
            # State C: perception weak + correct -> difficulty-weighted penalty
            r_len = -((1.0 + d_i) / 2.0) * float(np.clip(rho - 1.0, 0.0, 1.0))
        else:
            # State D: perception weak + wrong, or MAR unavailable -> zero gradient
            r_len = 0.0

        # R_perc (perception incentive/penalty)
        r_perc = 0.0
        if mar is not None and tau_mar is not None:
            mar_normalized = (mar - mar_min) / mar_range
            if mar > tau_mar and acc == 1.0:
                r_perc = GAMMA1 * mar_normalized
            elif mar <= tau_mar and acc == 1.0:
                r_perc = -GAMMA2 * ((tau_mar - mar_min) / mar_range)

        # total reward
        overall = r_ans + format_weight * fmt + len_weight * r_len + perc_weight * r_perc
        
        # Add a tiny global length penalty to prevent natural verbosity drift in PPO
        overall -= 0.01 * (length / 4096.0)

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
