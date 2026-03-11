#!/usr/bin/env python3
"""
PerceptGate 实验 wandb 数据分析脚本

用法:
    export WANDB_API_KEY="your_key_here"
    python scripts/analyze_wandb.py

会自动拉取 perceptgate-mar 和 fast-grpo 两个 project 下的所有 run，
生成对比图表和汇总 CSV。
"""

import os
import sys
from collections import defaultdict

import pandas as pd

try:
    import wandb
except ImportError:
    print("请先安装 wandb: pip install wandb")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib 未安装，跳过绘图。pip install matplotlib")


# ─── 配置 ──────────────────────────────────────────────────────────────────────
PROJECTS = ["perceptgate-mar", "fast-grpo"]
OUTPUT_DIR = "analysis_output"

# 实验名 → 显示标签（用于图例）
EXPERIMENT_LABELS = {
    "grpo-baseline": "E0: GRPO Baseline",
    "fg-step3-3b-baseline": "E1: FAST-GRPO Baseline",
    "pg-curriculum-ans-len-v2": "E3: Curriculum+R_ans+R_len",
    "pg-curriculum-ans-perc": "E4: Curriculum+R_ans+R_perc",
    "pg-curriculum-full-v2": "E5: PerceptGate Full",
    "pg-len_group-curriculum": "PG (len_group+curriculum)",
}

# 关注的核心指标
TRAIN_KEYS = [
    "reward/overall", "reward/accuracy", "reward/format",
    "reward/r_len", "reward/r_perc", "reward/mar",
    "actor/kl_coef", "actor/kl_penalty",
    "actor/policy_loss", "actor/entropy",
    "timing/step",
]

VAL_KEY_PATTERNS = ["val/", "reward_score"]
# ──────────────────────────────────────────────────────────────────────────────


def fetch_all_runs(api: wandb.Api, entity: str = None) -> dict[str, list]:
    """从所有 project 拉取 run 列表，按 experiment_name 分组"""
    runs_by_name = defaultdict(list)

    for project in PROJECTS:
        path = f"{entity}/{project}" if entity else project
        try:
            runs = api.runs(path)
            print(f"  [{project}] 找到 {len(runs)} 个 run")
            for run in runs:
                exp_name = run.config.get("trainer", {}).get("experiment_name", run.name)
                runs_by_name[exp_name].append(run)
        except Exception as e:
            print(f"  [{project}] 拉取失败: {e}")

    return dict(runs_by_name)


def fetch_run_history(run, keys=None, full=True) -> pd.DataFrame:
    """拉取单个 run 的完整 history"""
    if full:
        rows = list(run.scan_history(keys=keys))
    else:
        rows = list(run.history(keys=keys, samples=5000))
    df = pd.DataFrame(rows)
    return df


def print_summary_table(runs_by_name: dict):
    """打印所有实验的 summary 汇总"""
    print("\n" + "=" * 100)
    print("实验汇总表")
    print("=" * 100)

    records = []
    for exp_name, runs in sorted(runs_by_name.items()):
        for run in runs:
            s = run.summary
            record = {
                "experiment": exp_name,
                "run_id": run.id,
                "state": run.state,
                "steps": s.get("_step", "?"),
            }
            # 训练指标
            for key in ["reward/overall", "reward/accuracy", "reward/format",
                        "reward/r_len", "reward/r_perc", "reward/mar",
                        "actor/kl_coef"]:
                record[key] = s.get(key, None)

            # 验证指标 - 尝试常见 pattern
            for key in s.keys():
                if "val/" in key and ("accuracy" in key or "reward_score" in key or "response_length" in key):
                    record[key] = s.get(key)

            records.append(record)

    df = pd.DataFrame(records)
    print(df.to_string(index=False, max_colwidth=30))
    return df


def plot_training_curves(all_histories: dict[str, pd.DataFrame]):
    """绘制训练曲线对比图"""
    if not HAS_MPL:
        return

    # 核心对比指标
    plot_keys = [
        ("reward/accuracy", "Train Accuracy"),
        ("reward/overall", "Train Overall Reward"),
        ("reward/r_len", "R_len"),
        ("reward/r_perc", "R_perc"),
        ("reward/mar", "MAR"),
        ("actor/kl_coef", "KL Coefficient"),
        ("actor/entropy", "Policy Entropy"),
    ]

    # 过滤掉没有数据的指标
    available_keys = []
    for key, title in plot_keys:
        for df in all_histories.values():
            if key in df.columns and df[key].notna().any():
                available_keys.append((key, title))
                break

    if not available_keys:
        print("[WARN] 没有可绘制的训练指标")
        return

    n_plots = len(available_keys)
    n_cols = 2
    n_rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]
    axes = [ax for row in axes for ax in (row if hasattr(row, '__iter__') else [row])]

    colors = plt.cm.tab10.colors

    for idx, (key, title) in enumerate(available_keys):
        ax = axes[idx]
        for i, (exp_name, df) in enumerate(sorted(all_histories.items())):
            if key not in df.columns:
                continue
            subset = df[["_step", key]].dropna()
            if subset.empty:
                continue
            label = EXPERIMENT_LABELS.get(exp_name, exp_name)
            ax.plot(subset["_step"], subset[key], label=label,
                    color=colors[i % len(colors)], alpha=0.8, linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Step")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    # 隐藏多余的 subplot
    for idx in range(len(available_keys), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  训练曲线已保存: {path}")


def plot_val_curves(all_histories: dict[str, pd.DataFrame]):
    """绘制验证曲线对比图"""
    if not HAS_MPL:
        return

    # 收集所有 val 相关的 key
    val_keys = set()
    for df in all_histories.values():
        for col in df.columns:
            if col.startswith("val/"):
                val_keys.add(col)

    if not val_keys:
        print("[WARN] 没有验证指标数据")
        return

    # 按类型分组
    acc_keys = sorted(k for k in val_keys if "accuracy" in k)
    len_keys = sorted(k for k in val_keys if "response_length" in k or "length" in k)
    score_keys = sorted(k for k in val_keys if "reward_score" in k)

    groups = []
    if acc_keys:
        groups.append((acc_keys[0], "Val Accuracy"))
    if len_keys:
        groups.append((len_keys[0], "Val Response Length"))
    if score_keys:
        groups.append((score_keys[0], "Val Reward Score"))

    if not groups:
        # fallback: plot all val keys
        groups = [(k, k) for k in sorted(val_keys)[:4]]

    n_plots = len(groups)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors

    for idx, (key, title) in enumerate(groups):
        ax = axes[idx]
        for i, (exp_name, df) in enumerate(sorted(all_histories.items())):
            if key not in df.columns:
                continue
            subset = df[["_step", key]].dropna()
            if subset.empty:
                continue
            label = EXPERIMENT_LABELS.get(exp_name, exp_name)
            ax.plot(subset["_step"], subset[key], "o-", label=label,
                    color=colors[i % len(colors)], alpha=0.8, markersize=4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Step")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "val_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  验证曲线已保存: {path}")


def export_csv(all_histories: dict[str, pd.DataFrame]):
    """导出每个实验的完整数据为 CSV"""
    for exp_name, df in all_histories.items():
        safe_name = exp_name.replace("/", "_").replace(" ", "_")
        path = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")
        df.to_csv(path, index=False)
        print(f"  {exp_name}: {len(df)} rows → {path}")


def main():
    if not os.environ.get("WANDB_API_KEY"):
        print("请设置环境变量: export WANDB_API_KEY='your_key'")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    api = wandb.Api()

    # 尝试获取 entity
    try:
        entity = api.default_entity
        print(f"wandb entity: {entity}")
    except Exception:
        entity = None
        print("未检测到默认 entity，将尝试不带 entity 查询")

    # 1. 拉取所有 run
    print("\n[1/4] 拉取 wandb runs...")
    runs_by_name = fetch_all_runs(api, entity)

    if not runs_by_name:
        print("未找到任何 run。请检查 project 名称和 API key。")
        # 尝试列出用户的所有 project
        if entity:
            try:
                projects = api.projects(entity)
                print(f"\n你的 wandb 账户下有以下 project:")
                for p in projects:
                    print(f"  - {p.name}")
            except Exception:
                pass
        sys.exit(1)

    print(f"\n找到 {len(runs_by_name)} 个实验:")
    for name, runs in sorted(runs_by_name.items()):
        states = [r.state for r in runs]
        print(f"  {name}: {len(runs)} run(s), states={states}")

    # 2. 打印 summary 汇总
    print("\n[2/4] 汇总 summary...")
    summary_df = print_summary_table(runs_by_name)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "experiment_summary.csv"), index=False)

    # 3. 拉取完整 history
    print("\n[3/4] 拉取训练 history（可能需要几分钟）...")
    all_histories = {}
    for exp_name, runs in runs_by_name.items():
        # 取最新的 run（如果有多个）
        run = sorted(runs, key=lambda r: r.created_at, reverse=True)[0]
        print(f"  拉取 {exp_name} (run={run.id}, state={run.state})...")
        try:
            df = fetch_run_history(run)
            if not df.empty:
                all_histories[exp_name] = df
                print(f"    → {len(df)} rows, columns: {list(df.columns)[:10]}...")
        except Exception as e:
            print(f"    → 失败: {e}")

    # 4. 绘图 + 导出
    print("\n[4/4] 生成图表和 CSV...")
    if all_histories:
        plot_training_curves(all_histories)
        plot_val_curves(all_histories)
        export_csv(all_histories)
    else:
        print("没有可用的 history 数据")

    print(f"\n分析完成，输出目录: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
