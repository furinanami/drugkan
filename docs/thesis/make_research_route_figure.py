#!/usr/bin/env python3
"""Generate the research-route figure used in Chapter 1."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "figures" / "research_route_overview.png"
FONT_FILE = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def setup_font() -> None:
    if Path(FONT_FILE).exists():
        font_manager.fontManager.addfont(FONT_FILE)
        font_name = font_manager.FontProperties(fname=FONT_FILE).get_name()
    else:
        font_name = "DejaVu Sans"
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_name, "DejaVu Sans"],
        "axes.unicode_minus": False,
    })


def box(ax, xy, wh, title, body, fc, ec):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.6,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=12.5, weight="bold", color="#1f2933")
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center",
            fontsize=9.5, color="#36454f", linespacing=1.35)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.5,
        color="#4b5563",
        shrinkA=8,
        shrinkB=8,
    ))


def main() -> None:
    setup_font()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13.5, 5.4), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.93,
        "本文研究技术路线",
        ha="center",
        va="center",
        fontsize=18,
        weight="bold",
        color="#111827",
    )
    ax.text(
        0.5, 0.875,
        "从数据构建、模型放置消融到泛化评价与可解释性分析的完整流程",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#4b5563",
    )

    colors = [
        ("#e8f1fb", "#4a78b8"),
        ("#eef7ea", "#4f9b63"),
        ("#fff3dc", "#d9862f"),
        ("#f4ebfa", "#8758b4"),
        ("#e4f4f6", "#29899a"),
    ]
    items = [
        ("数据构建", "DrugComb 样本\n分子图 + 细胞表达\n协同标签 / Loewe 分数"),
        ("结构假设", "药物 A、药物 B、细胞系\n构成三元关系\n避免无结构拼接"),
        ("模型消融", "MLP baseline\nKAN head / DrugKAN\nHgKAN-HG / HgKAN-Agg"),
        ("泛化评价", "random split 检查插值\ncold-drug split 检查新药泛化\nAUPR / AUC / RMSE / MAE"),
        ("解释验证", "KAN 函数曲线\n药物扰动与保留响应\n原子 saliency"),
    ]

    x0, y0, w, h, gap = 0.035, 0.42, 0.16, 0.28, 0.04
    centers = []
    for idx, ((title, body), (fc, ec)) in enumerate(zip(items, colors)):
        x = x0 + idx * (w + gap)
        box(ax, (x, y0), (w, h), title, body, fc, ec)
        centers.append((x + w / 2, y0 + h / 2))

    for idx in range(len(centers) - 1):
        arrow(ax, (centers[idx][0] + w / 2, centers[idx][1]), (centers[idx + 1][0] - w / 2, centers[idx + 1][1]))

    ax.text(
        0.5, 0.22,
        "核心判断：KAN 的有效性取决于放置位置；结构化消息传递比直接替换预测头更适合药物协同预测。",
        ha="center",
        va="center",
        fontsize=11.5,
        color="#111827",
        weight="bold",
    )
    ax.plot([0.13, 0.87], [0.18, 0.18], color="#9ca3af", linewidth=1.0)
    ax.text(
        0.5, 0.12,
        "输出：性能主表、随机种子稳定性分析、案例级解释与局限性讨论",
        ha="center",
        va="center",
        fontsize=10.2,
        color="#4b5563",
    )

    fig.savefig(OUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
