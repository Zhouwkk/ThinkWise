#!/usr/bin/env python3
"""E4: ViRL39K (filtered) vs PAPO-Eval test overlap check for MM rebuttal."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

INSTRUCTION_SUFFIX = re.compile(
    r"\n?\s*You first think through the reasoning process.*?\\boxed\{\}\.\s*$",
    re.DOTALL | re.IGNORECASE,
)


def normalize_question(text: str) -> str:
    text = re.sub(r"<image>", "", text, flags=re.IGNORECASE)
    text = INSTRUCTION_SUFFIX.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_train_records(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for i, row in enumerate(data):
        q = normalize_question(row.get("question", ""))
        img_rel = row["image"][0] if isinstance(row.get("image"), list) else row.get("image", "")
        records.append(
            {
                "idx": i,
                "question_norm": q,
                "answer": (row.get("answer") or "").strip(),
                "source": row.get("source", ""),
                "qid": row.get("qid", ""),
                "image_rel": img_rel,
            }
        )
    return records


def load_eval_records(path: Path, papo_root: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = []
    for i, item in enumerate(data):
        question, answer = "", ""
        for msg in item.get("messages") or []:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role == "user":
                question = normalize_question(content)
            elif role == "assistant":
                answer = content
        images = item.get("images") or []
        img_rel = images[0] if images else ""
        if img_rel.startswith("./"):
            img_rel = img_rel[2:]
        records.append(
            {
                "idx": i,
                "question_norm": question,
                "answer": answer.strip(),
                "image_rel": img_rel,
                "image_abs": str((papo_root / img_rel).resolve()) if img_rel else "",
            }
        )
    return records


def token_set_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return SequenceMatcher(None, a, b).ratio()
    inter = len(ta & tb)
    return 2.0 * inter / (len(ta) + len(tb))


def file_md5(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_phash(path: Path) -> str | None:
    try:
        from PIL import Image
        import imagehash
    except ImportError:
        return None
    if not path.is_file():
        return None
    with Image.open(path) as im:
        return str(imagehash.phash(im.convert("RGB")))


def build_near_dup_index(train_records: list[dict[str, Any]], threshold: float) -> dict[int, list[int]]:
    """Map eval-like questions: bucket by length decile + first 40 chars."""
    buckets: dict[tuple[int, str], list[int]] = defaultdict(list)
    for rec in train_records:
        q = rec["question_norm"]
        if not q:
            continue
        key = (len(q) // 50, q[:40])
        buckets[key].append(rec["idx"])

    return buckets, threshold


def find_near_dup_train_idx(
    eval_q: str,
    train_records: list[dict[str, Any]],
    buckets: dict[tuple[int, str], list[int]],
    threshold: float,
) -> tuple[int | None, float]:
    if not eval_q:
        return None, 0.0
    candidates: set[int] = set()
    base_key = len(eval_q) // 50
    prefix = eval_q[:40]
    for delta in (-1, 0, 1):
        candidates.update(buckets.get((base_key + delta, prefix), []))
    if len(candidates) < 5:
        for delta in (-2, 2):
            candidates.update(buckets.get((base_key + delta, prefix), []))

    best_idx, best_score = None, 0.0
    for ti in candidates:
        score = token_set_ratio(eval_q, train_records[ti]["question_norm"])
        if score > best_score:
            best_score = score
            best_idx = ti
    if best_score >= threshold:
        return best_idx, best_score
    return None, best_score


BENCHMARKS: list[tuple[str, str]] = [
    ("MathVista", "AI4Math_MathVista.json"),
    ("MathVerse", "AI4Math_MathVerse.json"),
    ("MathVerse_v", "AI4Math_MathVerse_vision_dependent.json"),
    ("Counting", "BUAADreamer_clevr_count_70k.json"),
    ("MMMU-Pro", "MMMU_MMMU_Pro.json"),
    ("MathVision", "MathLLMs_MathVision.json"),
    ("MMK12", "PAPO_MMK12.json"),
    ("We-Math", "We_Math.json"),
    ("Geo3k", "hiyouga_geometry3k.json"),
    ("LogicVista", "lscpku_LogicVista.json"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="E4 contamination check")
    parser.add_argument(
        "--train_json",
        type=Path,
        default=Path("/data/zhouwenkang/FAST/train_data/ViRL39K/virl39k_filtered.json"),
    )
    parser.add_argument(
        "--train_image_root",
        type=Path,
        default=Path("/data/zhouwenkang/FAST/train_data/ViRL39K"),
    )
    parser.add_argument(
        "--papo_root",
        type=Path,
        default=Path("/data/zhouwenkang/PAPO-Eval"),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/data/zhouwenkang/FAST/experiments/e4"),
    )
    parser.add_argument("--near_dup_threshold", type=float, default=0.92)
    parser.add_argument("--skip_images", action="store_true")
    parser.add_argument("--phash", action="store_true", help="Also run perceptual hash (needs imagehash)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading train: {args.train_json}", flush=True)
    train_records = load_train_records(args.train_json)
    train_q_set = {r["question_norm"] for r in train_records if r["question_norm"]}
    q_to_train_idxs: dict[str, list[int]] = defaultdict(list)
    for r in train_records:
        if r["question_norm"]:
            q_to_train_idxs[r["question_norm"]].append(r["idx"])

    buckets, threshold = build_near_dup_index(train_records, args.near_dup_threshold)

    train_md5: dict[str, list[int]] = defaultdict(list)
    train_phash: dict[str, list[int]] = defaultdict(list)
    missing_train_images = 0
    if not args.skip_images:
        print(f"Hashing {len(train_records)} train images...", flush=True)
        for r in train_records:
            if not r["image_rel"]:
                continue
            p = args.train_image_root / r["image_rel"]
            md5 = file_md5(p)
            if md5 is None:
                missing_train_images += 1
                continue
            train_md5[md5].append(r["idx"])
            if args.phash:
                ph = file_phash(p)
                if ph:
                    train_phash[ph].append(r["idx"])

    per_benchmark: dict[str, Any] = {}
    details_path = args.out_dir / "overlap_details.jsonl"
    n_details = 0
    with open(details_path, "w", encoding="utf-8") as detail_f:
        for display_name, filename in BENCHMARKS:
            eval_path = args.papo_root / "data" / "papo" / filename
            if not eval_path.is_file():
                print(f"[WARN] missing eval file: {eval_path}", file=sys.stderr)
                continue

            eval_records = load_eval_records(eval_path, args.papo_root)
            n_eval = len(eval_records)
            exact_hits: list[int] = []
            near_hits: list[dict[str, Any]] = []
            image_md5_hits: list[dict[str, Any]] = []
            image_phash_hits: list[dict[str, Any]] = []
            missing_eval_images = 0

            for er in eval_records:
                eq = er["question_norm"]
                if eq in train_q_set:
                    exact_hits.append(er["idx"])
                    for ti in q_to_train_idxs[eq]:
                        hit = {
                            "benchmark": display_name,
                            "overlap_type": "exact_text",
                            "eval_idx": er["idx"],
                            "train_idx": ti,
                            "train_source": train_records[ti]["source"],
                            "question_preview": eq[:200],
                        }
                        detail_f.write(json.dumps(hit, ensure_ascii=False) + "\n")
                        n_details += 1
                else:
                    ti, score = find_near_dup_train_idx(
                        eq, train_records, buckets, args.near_dup_threshold
                    )
                    if ti is not None:
                        near_hits.append({"eval_idx": er["idx"], "train_idx": ti, "score": score})
                        hit = {
                            "benchmark": display_name,
                            "overlap_type": "near_dup_text",
                            "eval_idx": er["idx"],
                            "train_idx": ti,
                            "score": score,
                            "train_source": train_records[ti]["source"],
                            "question_preview": eq[:200],
                        }
                        detail_f.write(json.dumps(hit, ensure_ascii=False) + "\n")
                        n_details += 1

                if args.skip_images or not er["image_abs"]:
                    continue
                ep = Path(er["image_abs"])
                md5 = file_md5(ep)
                if md5 is None:
                    missing_eval_images += 1
                    continue
                if md5 in train_md5:
                    for ti in train_md5[md5]:
                        image_md5_hits.append({"eval_idx": er["idx"], "train_idx": ti, "md5": md5})
                        hit = {
                            "benchmark": display_name,
                            "overlap_type": "image_md5",
                            "eval_idx": er["idx"],
                            "train_idx": ti,
                            "train_source": train_records[ti]["source"],
                            "md5": md5,
                        }
                        detail_f.write(json.dumps(hit, ensure_ascii=False) + "\n")
                        n_details += 1
                if args.phash:
                    ph = file_phash(ep)
                    if ph and ph in train_phash:
                        for ti in train_phash[ph]:
                            image_phash_hits.append(
                                {"eval_idx": er["idx"], "train_idx": ti, "phash": ph}
                            )

            exact_unique = len(set(exact_hits))
            near_unique = len({h["eval_idx"] for h in near_hits})
            img_unique = len({h["eval_idx"] for h in image_md5_hits})

            per_benchmark[display_name] = {
                "eval_file": filename,
                "n_eval": n_eval,
                "exact_text_overlap_count": exact_unique,
                "exact_text_overlap_rate": round(exact_unique / n_eval, 6) if n_eval else 0.0,
                "near_dup_text_overlap_count": near_unique,
                "near_dup_text_overlap_rate": round(near_unique / n_eval, 6) if n_eval else 0.0,
                "near_dup_threshold": args.near_dup_threshold,
                "image_md5_overlap_count": img_unique,
                "image_md5_overlap_rate": round(img_unique / n_eval, 6) if n_eval else 0.0,
                "missing_eval_images": missing_eval_images,
            }
            if args.phash:
                per_benchmark[display_name]["image_phash_overlap_count"] = len(
                    {h["eval_idx"] for h in image_phash_hits}
                )

            print(
                f"{display_name:12s}  exact={exact_unique:4d}/{n_eval}  "
                f"near={near_unique:4d}  img_md5={img_unique:4d}",
                flush=True,
            )

    total_eval = sum(v["n_eval"] for v in per_benchmark.values())
    summary = {
        "train_json": str(args.train_json),
        "n_train": len(train_records),
        "papo_root": str(args.papo_root),
        "skip_images": args.skip_images,
        "missing_train_images": missing_train_images,
        "near_dup_threshold": args.near_dup_threshold,
        "per_benchmark": per_benchmark,
        "overall": {
            "n_eval_total": total_eval,
            "exact_text_overlap_count": sum(
                v["exact_text_overlap_count"] for v in per_benchmark.values()
            ),
            "near_dup_text_overlap_count": sum(
                v["near_dup_text_overlap_count"] for v in per_benchmark.values()
            ),
            "image_md5_overlap_count": sum(
                v["image_md5_overlap_count"] for v in per_benchmark.values()
            ),
        },
        "details_jsonl": str(details_path),
        "n_detail_rows": n_details,
    }
    if total_eval:
        summary["overall"]["exact_text_overlap_rate"] = round(
            summary["overall"]["exact_text_overlap_count"] / total_eval, 6
        )
        summary["overall"]["near_dup_text_overlap_rate"] = round(
            summary["overall"]["near_dup_text_overlap_count"] / total_eval, 6
        )
        summary["overall"]["image_md5_overlap_rate"] = round(
            summary["overall"]["image_md5_overlap_count"] / total_eval, 6
        )

    out_summary = args.out_dir / "e4_summary.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_summary}", flush=True)
    print(f"Wrote {details_path} ({n_details} rows)", flush=True)


if __name__ == "__main__":
    main()
