#!/usr/bin/env python3
"""Export a small RefCOCO val slice to local json+jpg (V*Bench-like layout)."""

import ast
import json
import argparse
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def parse_bbox(b):
    if isinstance(b, str):
        b = ast.literal_eval(b.strip())
    return [float(v) for v in b]


def bbox_to_xywh(b, img_w, img_h):
    """Kangheng/refcoco bbox is [x1,y1,x2,y2] in pixel coords."""
    a, b1, c, d = parse_bbox(b)
    if c > a and d > b1 and c <= img_w and d <= img_h:
        x, y, w, h = a, b1, c - a, d - b1
    else:
        x, y, w, h = a, b1, c, d
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return [int(round(x)), int(round(y)), int(round(w)), int(round(h))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="/data/zhouwenkang/FAST/eval_datasets/refcoco_mini")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Kangheng/refcoco", split=args.split, streaming=True)
    n = 0
    for row in ds:
        qid = f"refcoco_{row['question_id']}"
        img: Image.Image = row["image"].convert("RGB")
        w, h = img.size
        xywh = bbox_to_xywh(row["bbox"], w, h)
        ref = row["question"].strip()
        meta = {
            "question_id": qid,
            "benchmark": "refcoco",
            "ref_expr": ref,
            "question": ref,
            "target_object": [ref],
            "bbox": [xywh],
            "task_type": "referring_expression_comprehension",
        }
        img.save(out / f"{qid}.jpg", quality=95)
        with open(out / f"{qid}.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        n += 1
        if n >= args.n:
            break
    print(f"Wrote {n} samples to {out}")


if __name__ == "__main__":
    main()
