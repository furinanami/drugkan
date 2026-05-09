#!/usr/bin/env python3
"""Generate the overall thesis architecture figure.

Pipeline displayed:
  Inputs (Drug A graph, Drug B graph, Cell expression)
    -> DrugKAN encoder (GCN neighborhood aggregation + KAN node update)
    -> Cell MLP encoder
    -> Interaction module (MLP baseline / HgKAN-HG / HgKAN-Agg)
    -> Prediction heads (synergy classification, Loewe regression)
    -> Explainability hooks
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import numpy as np


OUT = Path(__file__).resolve().parent / "figures" / "model_architecture_overview.png"

from matplotlib import font_manager

_CJK_FONT_FILE = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_CJK_FONT_FILE)
_CJK_NAME = font_manager.FontProperties(fname=_CJK_FONT_FILE).get_name()
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [_CJK_NAME, "DejaVu Sans"],
    "axes.titleweight": "bold",
    "axes.unicode_minus": False,
})


# color palette
C_INPUT = "#E8F1FB"
C_INPUT_EDGE = "#4A78B8"
C_DRUG_ENC = "#FFF1E0"
C_DRUG_ENC_EDGE = "#E38B3C"
C_CELL_ENC = "#EEF7EA"
C_CELL_ENC_EDGE = "#5FA65F"
C_INTERACT = "#F4EBFA"
C_INTERACT_EDGE = "#8B5FBF"
C_KAN = "#FCE3E9"
C_KAN_EDGE = "#C8436C"
C_HEAD = "#FFF9CC"
C_HEAD_EDGE = "#B59B1E"
C_EXPLAIN = "#E4F4F6"
C_EXPLAIN_EDGE = "#2E8B9B"
C_ARROW = "#555555"
C_KAN_HIGHLIGHT = "#C8436C"


def box(ax, x, y, w, h, facecolor, edgecolor, text, fontsize=9.5,
        weight="normal", radius=0.02, text_color="#1f1f1f", line_width=1.4,
        text_align="center", zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=line_width, zorder=zorder,
    )
    ax.add_patch(patch)
    ha = "center"
    tx = x + w / 2
    if text_align == "left":
        ha = "left"
        tx = x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center",
            fontsize=fontsize, color=text_color, weight=weight, zorder=zorder + 1)


def arrow(ax, x1, y1, x2, y2, color=C_ARROW, lw=1.6, style="-|>",
          mutation=14, connection="arc3,rad=0.0", zorder=1, alpha=1.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mutation,
        color=color, linewidth=lw,
        connectionstyle=connection, zorder=zorder, alpha=alpha,
    ))


def draw_molecule(ax, cx, cy, scale=0.035, color="#4A78B8", label=None):
    # a small cartoon drug molecule: hexagonal ring + a branch
    coords = np.array([
        [0.0, 1.0], [0.87, 0.5], [0.87, -0.5],
        [0.0, -1.0], [-0.87, -0.5], [-0.87, 0.5],
    ]) * scale
    coords[:, 0] += cx
    coords[:, 1] += cy
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    for a, b in edges:
        ax.plot([coords[a, 0], coords[b, 0]], [coords[a, 1], coords[b, 1]],
                color=color, lw=1.4, zorder=3)
    # branch atom
    branch = np.array([0.87, 0.5]) * scale + [cx, cy]
    branch2 = branch + np.array([scale * 0.9, scale * 0.3])
    ax.plot([branch[0], branch2[0]], [branch[1], branch2[1]],
            color=color, lw=1.4, zorder=3)
    # atoms as circles
    for p in coords:
        ax.add_patch(Circle((p[0], p[1]), scale * 0.12,
                            facecolor="white", edgecolor=color,
                            lw=1.2, zorder=4))
    ax.add_patch(Circle((branch2[0], branch2[1]), scale * 0.12,
                        facecolor="white", edgecolor=color,
                        lw=1.2, zorder=4))
    if label is not None:
        ax.text(cx, cy - scale * 1.55, label, ha="center", va="top",
                fontsize=9.5, color=color, weight="bold")


def draw_kan_curve(ax, cx, cy, w, h, color=C_KAN_HIGHLIGHT, n=60):
    # Miniature KAN basis curve: a smooth wave
    xs = np.linspace(-1, 1, n)
    ys = 0.45 * np.sin(2.6 * xs) + 0.25 * np.cos(1.3 * xs + 0.5)
    ys = ys / (np.max(np.abs(ys)) + 1e-6)
    xr = cx - w / 2 + (xs + 1) / 2 * w
    yr = cy + ys * h / 2 * 0.85
    # thin box background
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                           facecolor="white", edgecolor="#bbbbbb",
                           lw=0.8, zorder=3))
    ax.plot(xr, yr, color=color, lw=1.6, zorder=4)
    ax.plot([cx - w / 2, cx + w / 2], [cy, cy],
            color="#cfcfcf", lw=0.5, ls=":", zorder=3)


def draw_hypergraph(ax, cx, cy, r=0.05, color=C_INTERACT_EDGE):
    # Three-node hyperedge: Drug A, Drug B, Cell
    A = (cx - r * 1.4, cy + r * 0.8)
    B = (cx + r * 1.4, cy + r * 0.8)
    Cn = (cx, cy - r * 1.2)
    # shaded hyperedge region
    from matplotlib.patches import Polygon
    pts = np.array([
        [A[0] - r * 0.6, A[1] + r * 0.5],
        [B[0] + r * 0.6, B[1] + r * 0.5],
        [Cn[0] + r * 0.8, Cn[1] - r * 0.6],
        [Cn[0] - r * 0.8, Cn[1] - r * 0.6],
    ])
    ax.add_patch(Polygon(pts, closed=True, facecolor=color, alpha=0.18,
                         edgecolor=color, lw=1.2, zorder=3))
    for (px, py), lab, col in [
        (A, "A", "#4A78B8"), (B, "B", "#E38B3C"), (Cn, "C", "#5FA65F")
    ]:
        ax.add_patch(Circle((px, py), r * 0.35, facecolor=col,
                            edgecolor="white", lw=1.2, zorder=4))
        ax.text(px, py, lab, ha="center", va="center", fontsize=7.5,
                color="white", weight="bold", zorder=5)


def draw_aggregated_graph(ax, cx, cy, r=0.05, color=C_INTERACT_EDGE):
    # drug-pair node + cell node with an edge between them
    P = (cx - r * 1.0, cy + r * 0.5)
    Cn = (cx + r * 1.2, cy - r * 0.5)
    ax.plot([P[0], Cn[0]], [P[1], Cn[1]], color=color, lw=1.6, zorder=3)
    ax.add_patch(Circle((P[0], P[1]), r * 0.45,
                        facecolor="#B37AD8", edgecolor="white",
                        lw=1.2, zorder=4))
    ax.text(P[0], P[1], "A+B", ha="center", va="center",
            fontsize=7.0, color="white", weight="bold", zorder=5)
    ax.add_patch(Circle((Cn[0], Cn[1]), r * 0.38,
                        facecolor="#5FA65F", edgecolor="white",
                        lw=1.2, zorder=4))
    ax.text(Cn[0], Cn[1], "C", ha="center", va="center",
            fontsize=7.5, color="white", weight="bold", zorder=5)


def main():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(0.5, 0.965,
            "基于 KAN 与超图神经网络的药物组合协同效应预测整体架构",
            ha="center", va="center", fontsize=15, weight="bold",
            color="#1f1f1f")
    ax.text(0.5, 0.932,
            "Drug A / Drug B 分子图 + 细胞系表达  →  KAN 增强编码  →  三元交互模块  →  协同分类 / Loewe 回归",
            ha="center", va="center", fontsize=10.5, color="#555555")

    # ========== Column 1: Inputs ==========
    col1_x = 0.04
    col_w = 0.16

    ax.text(col1_x + col_w / 2, 0.885, "① 输入",
            ha="center", fontsize=11.5, weight="bold", color="#333333")

    # Drug A input
    box(ax, col1_x, 0.73, col_w, 0.12, C_INPUT, C_INPUT_EDGE, "", radius=0.015)
    draw_molecule(ax, col1_x + col_w / 2, 0.78, scale=0.028,
                  color="#4A78B8", label="Drug A 分子图 G_a")

    # Drug B input
    box(ax, col1_x, 0.58, col_w, 0.12, C_INPUT, C_INPUT_EDGE, "", radius=0.015)
    draw_molecule(ax, col1_x + col_w / 2, 0.63, scale=0.028,
                  color="#E38B3C", label="Drug B 分子图 G_b")

    # Cell input
    box(ax, col1_x, 0.40, col_w, 0.14, C_INPUT, C_INPUT_EDGE, "", radius=0.015)
    # draw a small bar-style expression vector
    bar_x0 = col1_x + 0.018
    bar_y = 0.465
    bar_w = col_w - 0.036
    for i, v in enumerate(np.array([0.6, 0.9, 0.3, 0.75, 0.4, 0.85, 0.55, 0.25])):
        bx = bar_x0 + i * (bar_w / 8)
        ax.add_patch(Rectangle((bx, bar_y), bar_w / 8 * 0.78, v * 0.04,
                               facecolor="#5FA65F", edgecolor="white",
                               lw=0.5, zorder=4))
    ax.text(col1_x + col_w / 2, 0.43,
            "细胞系基因表达 x_c", ha="center", va="center",
            fontsize=9.5, color="#5FA65F", weight="bold")

    # Data summary
    box(ax, col1_x, 0.22, col_w, 0.14, "#FAFAFA", "#CCCCCC",
        "", radius=0.015, line_width=1.0)
    ax.text(col1_x + col_w / 2, 0.325, "DrugComb 数据集",
            ha="center", va="center", fontsize=9.8, weight="bold",
            color="#333333")
    ax.text(col1_x + col_w / 2, 0.295,
            "160,228 组合  |  1,896 药物  |  156 细胞系",
            ha="center", va="center", fontsize=8.3, color="#555555")
    ax.text(col1_x + col_w / 2, 0.270,
            "正例率 ≈ 14.8%",
            ha="center", va="center", fontsize=8.3, color="#555555")
    ax.text(col1_x + col_w / 2, 0.240,
            "划分: random / cold-drug / cold-cell",
            ha="center", va="center", fontsize=8.3,
            color="#8B5FBF", weight="bold")

    # ========== Column 2: Encoders ==========
    col2_x = 0.245
    col2_w = 0.22

    ax.text(col2_x + col2_w / 2, 0.885, "② 编码器",
            ha="center", fontsize=11.5, weight="bold", color="#333333")

    # DrugKAN encoder (shared) — spans drug A + B region
    box(ax, col2_x, 0.54, col2_w, 0.31,
        C_DRUG_ENC, C_DRUG_ENC_EDGE,
        "", radius=0.02, line_width=1.6)
    ax.text(col2_x + col2_w / 2, 0.825,
            "DrugKAN 分子图编码器  (5 层 KAGCN)",
            ha="center", va="center", fontsize=10.3, weight="bold",
            color=C_DRUG_ENC_EDGE)
    # inner formula
    ax.text(col2_x + col2_w / 2, 0.795,
            r"$\bar h_i = \frac{1}{2}(h_i + \mathrm{mean}_{j\in\mathcal{N}(i)} h_j)$",
            ha="center", va="center", fontsize=9.5, color="#333333")
    ax.text(col2_x + col2_w / 2, 0.765,
            r"$h_i' = \mathrm{skip}(h_i) + \mathrm{KAN}(\bar h_i)$",
            ha="center", va="center", fontsize=9.5, color=C_KAN_HIGHLIGHT,
            weight="bold")
    # KAN curve mini
    draw_kan_curve(ax, col2_x + 0.05, 0.685, 0.06, 0.04)
    ax.text(col2_x + 0.05, 0.648, "KAN 边函数",
            ha="center", va="center", fontsize=7.8, color=C_KAN_HIGHLIGHT)
    # aggregation cartoon
    ax_cx = col2_x + col2_w - 0.075
    ax_cy = 0.67
    # central node + neighbors
    ax.add_patch(Circle((ax_cx, ax_cy), 0.01,
                        facecolor=C_DRUG_ENC_EDGE, edgecolor="white",
                        lw=1.0, zorder=5))
    for dx, dy in [(-0.04, 0.02), (-0.035, -0.03), (0.035, -0.025), (0.04, 0.015)]:
        nx, ny = ax_cx + dx, ax_cy + dy
        ax.plot([ax_cx, nx], [ax_cy, ny], color=C_DRUG_ENC_EDGE,
                lw=0.9, zorder=4)
        ax.add_patch(Circle((nx, ny), 0.006,
                            facecolor="white", edgecolor=C_DRUG_ENC_EDGE,
                            lw=0.9, zorder=5))
    ax.text(ax_cx, ax_cy - 0.055, "邻域聚合",
            ha="center", va="top", fontsize=7.8, color="#555555")

    # outputs h_A, h_B
    ax.text(col2_x + col2_w / 2, 0.570,
            r"→ 药物嵌入  $h_A,\ h_B \in \mathbb{R}^{128}$",
            ha="center", va="center", fontsize=9.0, color="#333333",
            style="italic")

    # Cell encoder
    box(ax, col2_x, 0.40, col2_w, 0.14,
        C_CELL_ENC, C_CELL_ENC_EDGE, "", radius=0.02)
    ax.text(col2_x + col2_w / 2, 0.505, "Cell MLP 编码器",
            ha="center", va="center", fontsize=10.3, weight="bold",
            color=C_CELL_ENC_EDGE)
    ax.text(col2_x + col2_w / 2, 0.475,
            "两层全连接 + 非线性",
            ha="center", va="center", fontsize=9.0, color="#333333")
    ax.text(col2_x + col2_w / 2, 0.440,
            r"→ 细胞嵌入  $h_C \in \mathbb{R}^{128}$",
            ha="center", va="center", fontsize=9.0, color="#333333",
            style="italic")

    # KAN type note
    box(ax, col2_x, 0.225, col2_w, 0.15, "#FFFDF5", "#D4B94E",
        "", radius=0.015, line_width=1.0)
    ax.text(col2_x + col2_w / 2, 0.350,
            "KAN 函数层类型",
            ha="center", va="center", fontsize=9.8, weight="bold",
            color="#8A7815")
    ax.text(col2_x + col2_w / 2, 0.318,
            r"Fourier: $\phi_{o,i}(x)=a_0+\sum_k a_k\cos(k\pi x)+b_k\sin(k\pi x)$",
            ha="center", va="center", fontsize=8.2, color="#333333")
    ax.text(col2_x + col2_w / 2, 0.290,
            "B-spline: 分段局部平滑基函数",
            ha="center", va="center", fontsize=8.5, color="#333333")
    ax.text(col2_x + col2_w / 2, 0.258,
            "主实验采用 Fourier（cold-drug 更稳定）",
            ha="center", va="center", fontsize=8.2,
            color="#B5701E", weight="bold")

    # ========== Column 3: Interaction modules ==========
    col3_x = 0.495
    col3_w = 0.26

    ax.text(col3_x + col3_w / 2, 0.885, "③ Drug–Drug–Cell 交互模块",
            ha="center", fontsize=11.5, weight="bold", color="#333333")

    # Three stacked interaction variants
    var_h = 0.18
    var_gap = 0.025
    var_y_top = 0.68

    # Variant A: MLP baseline
    y1 = var_y_top
    box(ax, col3_x, y1, col3_w, var_h, "#F4F4F4", "#888888",
        "", radius=0.02)
    ax.text(col3_x + 0.012, y1 + var_h - 0.022, "(a) MLP baseline",
            ha="left", va="center", fontsize=10.0, weight="bold",
            color="#333333")
    ax.text(col3_x + 0.012, y1 + var_h - 0.045,
            r"$[\,h_A \parallel h_B \parallel h_C\,]\ \to\ \mathrm{MLP}$",
            ha="left", va="center", fontsize=9.2, color="#333333")
    # illustration: three small squares concat -> box
    bx0 = col3_x + col3_w - 0.105
    by0 = y1 + 0.035
    for i, col in enumerate(["#4A78B8", "#E38B3C", "#5FA65F"]):
        ax.add_patch(Rectangle((bx0 + i * 0.022, by0), 0.018, 0.028,
                               facecolor=col, edgecolor="white",
                               lw=1.0, zorder=4))
    arrow(ax, bx0 + 0.066 + 0.003, by0 + 0.014,
          bx0 + 0.096, by0 + 0.014, color="#666666", lw=1.2, mutation=10)
    ax.add_patch(Rectangle((bx0 + 0.098, by0), 0.018, 0.028,
                           facecolor="#BBBBBB", edgecolor="white",
                           lw=1.0, zorder=4))

    # Variant B: HgKAN-HG
    y2 = y1 - var_h - var_gap
    box(ax, col3_x, y2, col3_w, var_h, C_INTERACT, C_INTERACT_EDGE,
        "", radius=0.02, line_width=1.6)
    ax.text(col3_x + 0.012, y2 + var_h - 0.022,
            "(b) HgKAN-HG  (drug–drug–cell 超边)",
            ha="left", va="center", fontsize=10.0, weight="bold",
            color=C_INTERACT_EDGE)
    ax.text(col3_x + 0.012, y2 + var_h - 0.048,
            "node → hyperedge → node，两阶段 KAN 消息传递",
            ha="left", va="center", fontsize=8.8, color="#333333")
    ax.text(col3_x + 0.012, y2 + 0.020,
            "· 显式三元关系 · 解释主路径 (案例研究模型)",
            ha="left", va="center", fontsize=8.3, color="#6A4592",
            style="italic")
    # hypergraph icon
    draw_hypergraph(ax, col3_x + col3_w - 0.055, y2 + 0.08,
                    r=0.030, color=C_INTERACT_EDGE)

    # Variant C: HgKAN-Agg
    y3 = y2 - var_h - var_gap
    box(ax, col3_x, y3, col3_w, var_h, C_KAN, C_KAN_EDGE,
        "", radius=0.02, line_width=1.8)
    ax.text(col3_x + 0.012, y3 + var_h - 0.022,
            "(c) HgKAN-Agg  (药物对–细胞聚合图)   ★ 主性能路径",
            ha="left", va="center", fontsize=10.0, weight="bold",
            color=C_KAN_EDGE)
    ax.text(col3_x + 0.012, y3 + var_h - 0.048,
            r"$h_{AB}=\frac{1}{2}(h_A+h_B)\ \to\ \mathrm{KAN\text{-}MP}(h_{AB}, h_C)$",
            ha="left", va="center", fontsize=8.8, color="#333333")
    ax.text(col3_x + 0.012, y3 + 0.020,
            "· 先在药物对层面分组 · cold-drug 上同时优于 MLP 的分类与回归",
            ha="left", va="center", fontsize=8.3, color="#8B2F54",
            style="italic")
    # aggregated icon
    draw_aggregated_graph(ax, col3_x + col3_w - 0.055, y3 + 0.08,
                          r=0.030, color=C_KAN_EDGE)

    # ========== Column 4: Prediction + Explainability ==========
    col4_x = 0.79
    col4_w = 0.17

    ax.text(col4_x + col4_w / 2, 0.885, "④ 预测 & 解释",
            ha="center", fontsize=11.5, weight="bold", color="#333333")

    # Classification head
    box(ax, col4_x, 0.72, col4_w, 0.11, C_HEAD, C_HEAD_EDGE,
        "", radius=0.02)
    ax.text(col4_x + col4_w / 2, 0.785, "协同二分类",
            ha="center", va="center", fontsize=10.0, weight="bold",
            color="#8A7815")
    ax.text(col4_x + col4_w / 2, 0.750,
            "AUPR / AUC / F1",
            ha="center", va="center", fontsize=8.8, color="#333333")

    # Regression head
    box(ax, col4_x, 0.585, col4_w, 0.11, C_HEAD, C_HEAD_EDGE,
        "", radius=0.02)
    ax.text(col4_x + col4_w / 2, 0.650, "Loewe 分数回归",
            ha="center", va="center", fontsize=10.0, weight="bold",
            color="#8A7815")
    ax.text(col4_x + col4_w / 2, 0.615,
            r"RMSE / MAE / $R^2$",
            ha="center", va="center", fontsize=8.8, color="#333333")

    # Explainability box
    box(ax, col4_x, 0.225, col4_w, 0.33, C_EXPLAIN, C_EXPLAIN_EDGE,
        "", radius=0.02, line_width=1.4)
    ax.text(col4_x + col4_w / 2, 0.530, "可解释性工具链",
            ha="center", va="center", fontsize=10.3, weight="bold",
            color=C_EXPLAIN_EDGE)
    explain_items = [
        ("函数级", "KAN 边函数曲线"),
        ("样本级", "药物扰动 (mask A/B/both)"),
        ("样本级", "药物对表征保留响应 α∈[0,1]"),
        ("结构级", "原子 embedding saliency"),
        ("背景级", "细胞表达梯度 / 通路"),
    ]
    ey = 0.490
    for tag, name in explain_items:
        ax.add_patch(Rectangle((col4_x + 0.010, ey - 0.018), 0.035, 0.024,
                               facecolor=C_EXPLAIN_EDGE, edgecolor="none",
                               zorder=4))
        ax.text(col4_x + 0.0275, ey - 0.006, tag, ha="center", va="center",
                fontsize=7.2, color="white", weight="bold", zorder=5)
        ax.text(col4_x + 0.052, ey - 0.006, name, ha="left", va="center",
                fontsize=8.5, color="#1f1f1f")
        ey -= 0.048
    ax.text(col4_x + col4_w / 2, 0.248,
            "案例: Lenalidomide × mitomycin C\n@ IGROV1  (sample 57954)",
            ha="center", va="center", fontsize=8.3,
            color="#2E8B9B", style="italic")

    # ========== Arrows: input -> encoder ==========
    # Drug A/B -> DrugKAN
    arrow(ax, col1_x + col_w, 0.79, col2_x, 0.74,
          connection="arc3,rad=-0.10")
    arrow(ax, col1_x + col_w, 0.64, col2_x, 0.66,
          connection="arc3,rad=0.10")
    # Cell -> Cell MLP
    arrow(ax, col1_x + col_w, 0.47, col2_x, 0.47)

    # Encoder -> Interaction (fan-in)
    # h_A, h_B from DrugKAN -> each interaction box
    enc_out_x = col2_x + col2_w
    # to (a)
    arrow(ax, enc_out_x, 0.68, col3_x, y1 + var_h / 2,
          connection="arc3,rad=0.10")
    # to (b)
    arrow(ax, enc_out_x, 0.64, col3_x, y2 + var_h / 2,
          connection="arc3,rad=0.0")
    # to (c)
    arrow(ax, enc_out_x, 0.60, col3_x, y3 + var_h / 2,
          connection="arc3,rad=-0.10")
    # cell -> interaction variants
    arrow(ax, enc_out_x, 0.47, col3_x, y1 + var_h / 2 - 0.04,
          connection="arc3,rad=-0.25", color="#8BAF73", lw=1.3)
    arrow(ax, enc_out_x, 0.47, col3_x, y2 + var_h / 2 - 0.04,
          connection="arc3,rad=-0.10", color="#8BAF73", lw=1.3)
    arrow(ax, enc_out_x, 0.47, col3_x, y3 + var_h / 2,
          connection="arc3,rad=0.05", color="#8BAF73", lw=1.3)

    # Interaction -> heads (only from selected module representative arrows)
    int_out_x = col3_x + col3_w
    # to classification
    arrow(ax, int_out_x, y1 + var_h / 2, col4_x, 0.775,
          connection="arc3,rad=0.10", color="#777777", lw=1.3, alpha=0.6)
    arrow(ax, int_out_x, y2 + var_h / 2, col4_x, 0.775,
          connection="arc3,rad=0.05", color="#777777", lw=1.3, alpha=0.6)
    arrow(ax, int_out_x, y3 + var_h / 2, col4_x, 0.775,
          connection="arc3,rad=-0.15", color=C_KAN_EDGE, lw=1.8)
    # to regression
    arrow(ax, int_out_x, y1 + var_h / 2 - 0.02, col4_x, 0.640,
          connection="arc3,rad=0.20", color="#777777", lw=1.3, alpha=0.6)
    arrow(ax, int_out_x, y2 + var_h / 2 - 0.02, col4_x, 0.640,
          connection="arc3,rad=0.10", color="#777777", lw=1.3, alpha=0.6)
    arrow(ax, int_out_x, y3 + var_h / 2 - 0.02, col4_x, 0.640,
          connection="arc3,rad=-0.05", color=C_KAN_EDGE, lw=1.8)

    # Explainability dashed feedback arrows
    arrow(ax, col2_x + col2_w - 0.02, 0.685, col4_x, 0.45,
          color=C_EXPLAIN_EDGE, lw=1.1, mutation=10,
          style="-|>", connection="arc3,rad=-0.35", alpha=0.55)
    arrow(ax, col3_x + col3_w / 2, y3, col4_x, 0.36,
          color=C_EXPLAIN_EDGE, lw=1.1, mutation=10,
          style="-|>", connection="arc3,rad=-0.2", alpha=0.55)

    # ========== Footer: key finding ==========
    fbg = FancyBboxPatch((0.04, 0.05), 0.92, 0.13,
                         boxstyle="round,pad=0.01,rounding_size=0.015",
                         facecolor="#FAFBF5", edgecolor="#C9C28F",
                         linewidth=1.2, zorder=2)
    ax.add_patch(fbg)
    ax.text(0.06, 0.148, "核心设计原则",
            ha="left", va="center", fontsize=10.3, weight="bold",
            color="#6B601A")
    ax.text(0.06, 0.118,
            "图结构提供“分组 / 局部化”归纳偏置，KAN 提供“边上的可学习函数”；"
            "二者结合优于无结构地用 KAN 替换最终 MLP。",
            ha="left", va="center", fontsize=9.5, color="#333333")
    ax.text(0.06, 0.090,
            "★ cold-drug 场景下，HgKAN-Agg 是唯一同时在分类 (AUPR 0.2765, AUC 0.7452) "
            "与回归 (RMSE 17.68) 上优于 MLP baseline 的 KAN 主变体。",
            ha="left", va="center", fontsize=9.3, color="#8B2F54")
    ax.text(0.06, 0.064,
            "△ 直接用 KAN 替换预测头 (KAN-head) 在 random 与 cold-drug 上均弱于 MLP，说明 KAN 不适合拼接后的高维无结构输入。",
            ha="left", va="center", fontsize=9.0, color="#555555")

    # Legend
    legend_elements = [
        Line2D([0], [0], color=C_KAN_EDGE, lw=2.2, label="主性能/主解释路径"),
        Line2D([0], [0], color="#8BAF73", lw=1.6, label="细胞系特征流"),
        Line2D([0], [0], color=C_ARROW, lw=1.6, label="药物特征流"),
        Line2D([0], [0], color=C_EXPLAIN_EDGE, lw=1.4, ls="-",
               alpha=0.6, label="解释性分析接入点"),
    ]
    leg = ax.legend(handles=legend_elements, loc="lower right",
                    bbox_to_anchor=(0.965, 0.045),
                    fontsize=8.5, frameon=True, framealpha=0.95,
                    edgecolor="#C9C28F", facecolor="white")
    leg.get_frame().set_linewidth(1.0)

    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
