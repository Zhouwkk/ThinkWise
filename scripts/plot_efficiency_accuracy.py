import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullLocator

# =========================
# Data
# =========================
data_3b = [
    ("Curr-ReFT",    34.58, 112.5),
    ("RLCS",         44.04, 620.1),
    ("VLAA",         38.33, 499.0),
    ("FAST",         45.76, 220.4),
    ("MMR1",         40.49, 941.3),
    ("PAPO",         48.47, 315.0),
    ("ThinkWise-3B", 53.11, 225.3),
]

data_7b = [
    ("Curr-ReFT",    43.27, 155.4),
    ("RLCS",         34.19, 608.4),
    ("VLAA",         47.69, 304.9),
    ("FAST",         59.22, 155.2),
    ("MMR1",         40.23, 907.4),
    ("R1-Onevision", 34.00, 226.2),
    ("ReVisual-R1",  50.05, 2207.6),
    ("PAPO",         59.85, 314.7),
    ("ThinkWise-7B", 61.92, 302.7),
]

# =========================
# Colors
# =========================
COLOR_3B  = "#94a3b8"
COLOR_7B  = "#60a5fa"
COLOR_TW3 = "#f97316"
COLOR_TW7 = "#dc2626"

# =========================
# Split data
# =========================
base3 = [(n, a, l) for n, a, l in data_3b if not n.startswith("ThinkWise")]
tw3   = [(n, a, l) for n, a, l in data_3b if n.startswith("ThinkWise")]
base7 = [(n, a, l) for n, a, l in data_7b if not n.startswith("ThinkWise")]
tw7   = [(n, a, l) for n, a, l in data_7b if n.startswith("ThinkWise")]

# =========================
# Manual label offsets
# key format: (series, name) -> (dx, dy)
# dx, dy are in points
# =========================
label_offsets = {
    ("3B", "Curr-ReFT"):   (0, 10),
    ("3B", "RLCS"):        (0, 10),
    ("3B", "VLAA"):        (0, 10),
    ("3B", "FAST"):        (0, -14),
    ("3B", "MMR1"):        (18, 8),
    ("3B", "PAPO"):        (14, 8),
    ("3B", "ThinkWise-3B"): (0, 12),

    ("7B", "Curr-ReFT"):   (0, -14),
    ("7B", "RLCS"):        (0, 10),
    ("7B", "VLAA"):        (-14, 8),
    ("7B", "FAST"):        (0, 10),
    ("7B", "MMR1"):         (-18, 8),
    ("7B", "R1-Onevision"): (16, 8),
    ("7B", "ReVisual-R1"): (-10, 10),
    ("7B", "PAPO"):        (18, -2),
    ("7B", "ThinkWise-7B"): (0, 12),
}

# =========================
# Figure
# =========================
fig, ax = plt.subplots(figsize=(7.8, 5.2), dpi=300)

# Scatter points
ax.scatter(
    [l for _, a, l in base3],
    [a for _, a, l in base3],
    s=85, marker='o', color=COLOR_3B,
    edgecolors='white', linewidths=0.9,
    label='3B baselines', zorder=3
)

ax.scatter(
    [l for _, a, l in base7],
    [a for _, a, l in base7],
    s=85, marker='D', color=COLOR_7B,
    edgecolors='white', linewidths=0.9,
    label='7B baselines', zorder=3
)

ax.scatter(
    [l for _, a, l in tw3],
    [a for _, a, l in tw3],
    s=260, marker='*', color=COLOR_TW3,
    edgecolors='white', linewidths=1.0,
    label='ThinkWise-3B', zorder=5
)

ax.scatter(
    [l for _, a, l in tw7],
    [a for _, a, l in tw7],
    s=280, marker='*', color=COLOR_TW7,
    edgecolors='white', linewidths=1.0,
    label='ThinkWise-7B', zorder=6
)

# =========================
# Annotation helper
# =========================
def annotate_series(data, series, color, fontsize=9, fontweight='normal'):
    for name, acc, length in data:
        dx, dy = label_offsets.get((series, name), (0, 8))
        ax.annotate(
            name,
            xy=(length, acc),
            xytext=(dx, dy),
            textcoords="offset points",
            ha='center',
            va='bottom' if dy >= 0 else 'top',
            fontsize=fontsize,
            color=color,
            fontweight=fontweight,
            zorder=10,
        )

annotate_series(base3, "3B", COLOR_3B, fontsize=8.8)
annotate_series(base7, "7B", COLOR_7B, fontsize=8.8)
annotate_series(tw3, "3B", COLOR_TW3, fontsize=10, fontweight='bold')
annotate_series(tw7, "7B", COLOR_TW7, fontsize=10, fontweight='bold')

# =========================
# Axes
# =========================
ax.set_xscale('log')
ax.set_xlabel("Avg Response Length (log scale)", fontsize=12)
ax.set_ylabel("Avg Accuracy (%)", fontsize=12)

ax.set_xticks([100, 200, 500, 1000, 2000])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_locator(NullLocator())

ax.tick_params(axis='both', labelsize=10)

ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.35, zorder=0)
ax.set_xlim(90, 2600)
ax.set_ylim(32, 64)

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.14),
    ncol=4,
    frameon=False,
    fontsize=10,
    handletextpad=0.6,
    columnspacing=1.2
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("efficiency_accuracy_paper_v2.png", dpi=300, bbox_inches="tight")
plt.savefig("efficiency_accuracy_paper_v2.pdf", bbox_inches="tight")
plt.show()