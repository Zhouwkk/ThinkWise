#!/usr/bin/env bash
# E1: label 800 trajectories (extreme MAR tails only) with resume + retry playbook.
# Usage:
#   export SILICONFLOW_API_KEY="..."
#   conda activate fast_grpo
#   cd /data/zhouwenkang/FAST
#   bash experiments/e1/run_extreme_800.sh          # label (resumable)
#   bash experiments/e1/run_extreme_800.sh retry      # re-label API errors only
#   bash experiments/e1/run_extreme_800.sh analyze    # metrics after labeling

set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

LABEL_OUT="${ROOT}/experiments/e1/labeled_trajectories_extreme_800.jsonl"
LOG_LABEL="${ROOT}/experiments/e1/label_extreme_800.log"
LOG_RETRY="${ROOT}/experiments/e1/retry_extreme_800.log"
PY=(python pilot_experiment/scripts/e1_mar_perception_validation.py)
CONFIG="${E1_JUDGE_CONFIG:-experiments/e1/judge_config.yaml}"

if [[ -z "${SILICONFLOW_API_KEY:-}" ]]; then
  echo "ERROR: set SILICONFLOW_API_KEY before running." >&2
  exit 1
fi

export PYTHONUNBUFFERED=1

cmd="${1:-label}"

case "$cmd" in
  label)
    echo "=== Label 800 (extreme MAR tails), output: ${LABEL_OUT} ==="
    echo "Progress: tail -f ${LOG_LABEL}"
    echo "Count:    watch -n 60 'wc -l < ${LABEL_OUT} 2>/dev/null || echo 0'"
  echo "Using judge config: ${CONFIG}"
  "${PY[@]}" --phase label \
      --sample 800 \
      --extreme-mar-only \
      --config "${CONFIG}" \
      --labeled-out "${LABEL_OUT}" \
      --resume \
      2>&1 | tee -a "${LOG_LABEL}"
    ;;
  retry)
    echo "=== Retry API errors only ==="
    "${PY[@]}" --phase label \
      --sample 800 \
      --extreme-mar-only \
      --config "${CONFIG}" \
      --labeled-out "${LABEL_OUT}" \
      --retry-errors \
      2>&1 | tee -a "${LOG_RETRY}"
    ;;
  recover)
    echo "=== Recover labels from JSON embedded in error explanations (no API) ==="
    "${PY[@]}" --phase recover-parse --labeled-out "${LABEL_OUT}"
    ;;
  analyze)
    echo "=== Analyze ==="
    "${PY[@]}" --phase analyze --labeled-out "${LABEL_OUT}"
    "${PY[@]}" --phase analyze-ideal --labeled-out "${LABEL_OUT}"
    ;;
  status)
    python3 - <<'PY'
import json
from collections import Counter
from pathlib import Path
p = Path("experiments/e1/labeled_trajectories_extreme_800.jsonl")
if not p.exists():
    print("No file yet:", p)
    raise SystemExit(0)
rows = [json.loads(l) for l in p.open() if l.strip()]
c = Counter(r.get("vlm_judgment") for r in rows)
print("n=", len(rows), "/ 800 target")
print("judgments:", dict(c))
print("errors:", c.get("error", 0))
PY
    ;;
  *)
    echo "Usage: $0 {label|retry|recover|analyze|status}" >&2
    exit 1
    ;;
esac
