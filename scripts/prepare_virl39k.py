"""
将 ViRL39K parquet 转为 FAST-GRPO dataset.py 兼容的 JSON 格式。

处理内容:
  1. 补全缺少 <image> 占位符的 question（851/38870 条）
  2. 从 parquet 转为 JSON，避免 dataset.py 的 data_dir 模式加载报错
  3. 保留所有原始字段

用法:
  python scripts/prepare_virl39k.py \
      --input /data/zhouwenkang/FAST/train_data/ViRL39K/39Krelease.parquet \
      --output /data/zhouwenkang/FAST/train_data/ViRL39K/virl39k_train.json
"""

import json
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to 39Krelease.parquet")
    parser.add_argument("--output", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows from {args.input}")

    fixed_count = 0
    records = []
    for _, row in df.iterrows():
        entry = {}
        # question: 补全 <image>
        q = row["question"]
        if "<image>" not in q:
            q = "<image>\n" + q
            fixed_count += 1
        entry["question"] = q

        # answer: 已有 \boxed{}，直接保留
        entry["answer"] = row["answer"]

        # image: numpy array -> list
        img = row["image"]
        entry["image"] = list(img) if hasattr(img, "tolist") else list(img)

        # 保留元信息
        for key in ["category", "source", "qid"]:
            if key in row and pd.notna(row[key]):
                entry[key] = row[key]

        records.append(entry)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Done! Saved {len(records)} samples to {args.output}")
    print(f"Fixed {fixed_count} questions missing <image> tag")


if __name__ == "__main__":
    main()
