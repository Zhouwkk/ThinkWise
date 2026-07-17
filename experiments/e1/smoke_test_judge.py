#!/usr/bin/env python3
"""Smoke test: SiliconFlow Qwen3.5-397B-A17B as E1 VLM judge."""

import json
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pilot_experiment.verifiers.vlm_judge import JUDGE_SYSTEM_PROMPT, VLMJudgeVerifier


def main():
    cfg = OmegaConf.load(ROOT / "experiments/e1/judge_config.yaml")
    vcfg = cfg.verifiers.vlm_judge
    api_key = (
        vcfg.get("api_key")
        or os.environ.get("SILICONFLOW_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        print("Set SILICONFLOW_API_KEY before running.")
        sys.exit(1)

    merged = ROOT / "experiments/e1/merged_trajectories.jsonl"
    row = json.loads(merged.open().readline())

    # Same client pattern as SiliconFlow docs
    client = OpenAI(api_key=api_key, base_url=vcfg.api_base)
    print(f"model={vcfg.judge_model} base={vcfg.api_base}")

    judge = VLMJudgeVerifier(vcfg)
    result = judge.verify(row["thinking_text"], row["image_path"])
    print(f"judgment={result.judgment} grounded={result.is_grounded}")
    print(f"explanation={result.explanation}")


if __name__ == "__main__":
    main()
