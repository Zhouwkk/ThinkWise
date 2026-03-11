"""
将 FanqingM/MMK12 test split 预处理为 FAST-GRPO dataset.py 兼容的格式。

MMK12 原始格式 (HuggingFace):
  - question: str (题目文本)
  - image: PIL.Image (单张图片)
  - answer: str (如 "B")
  - options: list[str] (选项列表)
  - 其他字段: category, source 等

FAST-GRPO dataset.py 期望格式:
  - question: str (含 <image> 占位符)
  - answer: str (用 \boxed{} 包裹)
  - image: list[str] (图片文件名列表)

用法:
  python scripts/prepare_mmk12_val.py \
      --output_dir /data/zhouwenkang/FAST/val_data/MMK12 \
      --hf_token <your_token>  # 可选
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess MMK12 for FAST-GRPO validation")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for preprocessed data")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to download")
    args = parser.parse_args()

    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading FanqingM/MMK12 split={args.split} ...")
    dataset = load_dataset("FanqingM/MMK12", split=args.split, token=hf_token)
    print(f"Loaded {len(dataset)} samples")

    converted = []
    for idx, item in enumerate(dataset):
        # 1. 保存图片到磁盘
        image = item["image"]  # PIL.Image
        img_filename = f"mmk12_{idx:05d}.png"
        img_path = image_dir / img_filename
        image.save(img_path)

        # 2. 构造 question: 加 <image> 占位符 + 拼接选项
        question = item["question"]
        if "<image>" not in question:
            question = "<image>\n" + question

        # 拼接选项（如果有）
        if "options" in item and item["options"]:
            opts = item["options"]
            if isinstance(opts, list):
                opt_labels = "ABCDEFGH"
                opt_lines = []
                for i, opt in enumerate(opts):
                    if i < len(opt_labels):
                        opt_lines.append(f"{opt_labels[i]}. {opt}")
                if opt_lines:
                    question = question + "\n" + "\n".join(opt_lines)

        # 3. answer 用 \boxed{} 包裹
        answer = item.get("answer", "")
        boxed_answer = f"\\boxed{{{answer}}}"

        # 4. image 字段为列表，使用绝对路径
        #    这样即使 config 中 image_dir 指向训练集目录，
        #    os.path.join(image_dir, abs_path) 仍返回 abs_path
        entry = {
            "question": question,
            "answer": boxed_answer,
            "image": [str(img_path.resolve())],
        }

        # 保留元信息（不影响 dataset.py 加载，但方便分析）
        for meta_key in ["category", "source", "id"]:
            if meta_key in item:
                entry[meta_key] = item[meta_key]

        converted.append(entry)

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)}")

    # 保存为 JSON
    json_path = output_dir / "mmk12_test.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Saved {len(converted)} samples to {json_path}")
    print(f"Images saved to {image_dir}/")
    print(f"\nTo use in config_perceptgate.yaml:")
    print(f"  val_files:")
    print(f"    - {json_path}")
    print(f"  (image_dir for val is: {output_dir})")


if __name__ == "__main__":
    main()
