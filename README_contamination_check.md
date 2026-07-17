# Split-Aware Contamination Check

Checks whether **ViRL39K train** (`virl39k_train.json`) overlaps with the **exact evaluation splits** used in PerceptGate experiments (PAPO-Eval). Training-split overlap is reported separately and is **not** counted as evaluation leakage.

## Monitoring progress

While running, tail these files:

```bash
tail -f outputs/contamination_check/run.log
tail -f outputs/contamination_check/latest_status.txt
cat outputs/contamination_check/summary.partial.json
```

After each benchmark completes, incremental artifacts land in:

- `checkpoints/progress.json` — completed benchmark list + current phase
- `checkpoints/benchmarks/{Name}_overlap.csv` — per-benchmark results
- `overlap_candidates.csv`, `confirmed_eval_overlap.csv`, … — cumulative CSVs refreshed each benchmark
- `summary.partial.json` — running totals

Resume an interrupted run:

```bash
python scripts/check_contamination.py ... --resume
```

Image hashes are cached in `checkpoints/image_hash_cache.jsonl` so re-runs skip already-hashed files.

## Speed tips

| Bottleneck | Fix | Typical time (38,870 train) |
|------------|-----|----------------------------|
| **difflib near-dup** (no sklearn) | `conda run -n fast_grpo pip install scikit-learn` | days → **~30–45 min** full run |
| **Near-duplicate optional** | `--skip-near-duplicate` (ID + image hash still run) | **~10–20 min** |
| **Image hash only** | `--skip-near-duplicate --skip-images` | **~15 s** (ID + exact text) |
| **Parallel image hash** | `--hash-workers 8` (default: min(8, CPUs)) | ~4× faster hashing |
| **Re-run after interrupt** | `--resume` + existing `image_hash_cache.jsonl` | skips finished benchmarks / cached images |

The script builds **one shared TF-IDF index** on the full train set (not once per benchmark). Ensure `run.log` shows `Loaded 38870 train records` and `virl39k_train.json`, not `26962` / `virl39k_filtered.json`.


```bash
cd /data/zhouwenkang/FAST

# Smoke test (20 samples per split)
python scripts/check_contamination.py \
  --train-root /data/zhouwenkang/FAST/train_data/ViRL39K \
  --eval-root /data/zhouwenkang/PAPO-Eval \
  --config configs/eval_splits.yaml \
  --out outputs/contamination_check \
  --max-samples 20

# Full offline run
python scripts/check_contamination.py \
  --train-root /data/zhouwenkang/FAST/train_data/ViRL39K \
  --eval-root /data/zhouwenkang/PAPO-Eval \
  --config configs/eval_splits.yaml \
  --out outputs/contamination_check
```

## Outputs

| File | Description |
|------|-------------|
| `summary.md` | Human-readable report + rebuttal paragraph |
| `summary.json` | Machine-readable stats |
| `overlap_candidates.csv` | All non-benign candidates |
| `confirmed_eval_overlap.csv` | Split-aware confirmed eval leakage |
| `near_duplicate_candidates.csv` | High text similarity, not auto-confirmed |
| `image_overlap_candidates.csv` | SHA256 / pHash image hits |
| `filtered_eval_ids.json` | Eval IDs to drop for sensitivity re-eval (if any confirmed) |

## Principles

1. **Confirmed contamination** requires eval-split evidence (metadata test ID match, or strong text+image agreement).
2. **Benign overlap** = ViRL39K sample ID belongs to official **train** split of the same benchmark (when reference parquets are configured).
3. **Near-duplicates** and **image-only** hits go to manual review unless confirmation rules are met.
4. **No network** — all paths are local. Optional `test_id_parquet` / `train_id_parquet_glob` in `configs/eval_splits.yaml` point to cached HF parquet files for ID-level split awareness (MMK12, MathVision).

## Thresholds

```bash
--text-threshold 0.90        # TF-IDF / difflib near-duplicate
--phash-strong-threshold 5   # strong pHash match
--phash-weak-threshold 10    # weak pHash candidate
--topk 5                     # near-duplicate retrieval depth
```

## Config

Edit `configs/eval_splits.yaml` to add benchmarks, fix split names, or point to local official split ID parquets.

Train manifest defaults to `virl39k_train.json` under `--train-root` (38,870 samples; full ViRL39K train split used in GRPO).
