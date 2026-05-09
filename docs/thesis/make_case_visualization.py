#!/usr/bin/env python3
"""
Build visual interpretability figures for the thesis case study.

The selected case is sample 57954:
Lenalidomide + mitomycin C on IGROV1, a cold-drug test example.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import pandas as pd
from PIL import Image, ImageChops, ImageOps


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "thesis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY = (
    ROOT
    / "graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1/experiments/thesis_summary"
)
CASE_DIR = (
    SUMMARY
    / "explanations/samples/hgkan_only_hg_cold_pat30_tp_57954"
)
KAN_DIR = (
    SUMMARY
    / "explanations/kan_functions_hgkan_only_hg_cls_cold_drug_pat30"
)
AGG_KAN_DIR = (
    SUMMARY
    / "explanations/kan_functions_hgkan_only_agg_cls_cold_drug"
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#2b2b2b",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    }
)


def save_modality_plot(output_name="case_57954_modality_perturbation.png", include_title=True):
    df = pd.read_csv(CASE_DIR / "modality_perturbation.csv")
    order = ["none", "drug_a_zero", "drug_b_zero", "drug_pair_zero"]
    df = df[df["mask"].isin(order)].copy()
    df["mask"] = pd.Categorical(df["mask"], categories=order, ordered=True)
    df = df.sort_values("mask")

    label_map = {
        "none": "Original",
        "drug_a_zero": "Mask\nLenalidomide",
        "drug_b_zero": "Mask\nmitomycin C",
        "drug_pair_zero": "Mask\nboth drugs",
    }
    df["label"] = df["mask"].map(label_map)

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    colors = ["#2667a0", "#e08d3c", "#c94845", "#7f1d2d"]
    bars = ax.bar(df["label"], df["score"], color=colors, alpha=0.88)
    ax.axhline(0.5, color="#222222", linewidth=1.2, linestyle="--", alpha=0.75)
    ax.text(
        len(df) - 0.55,
        0.515,
        "decision threshold",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    ax.set_ylabel("Synergy probability")
    ax.set_ylim(0, 1.02)
    if include_title:
        ax.set_title("Drug perturbation controls the prediction")
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    for bar, value, delta in zip(bars, df["score"], df["delta_from_base"]):
        delta_label = "base" if abs(delta) < 1e-8 else f"{delta:+.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 0.98),
            f"{value:.3f}\n{delta_label}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#202020",
        )
    fig.tight_layout()
    path = OUT_DIR / output_name
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_gene_plot():
    df = pd.read_csv(CASE_DIR / "expression_gradient_importance.csv").head(14)
    expr = pd.read_csv(ROOT / "graphnn/data/preprocessing/df_rma_landm.tsv", sep="\t", index_col=0)
    gene_names = list(expr.index)
    df["gene_symbol"] = df["feature_index"].map(
        lambda idx: gene_names[int(idx)] if int(idx) < len(gene_names) else f"feature_{idx}"
    )

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    view = df.iloc[::-1]
    ax.barh(view["gene_symbol"], view["gradient_abs"], color="#497a5d", alpha=0.9)
    ax.set_xlabel("Absolute gradient")
    ax.set_title("Supplementary: expression gradients")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    path = OUT_DIR / "case_57954_expression_gradient.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def add_box(ax, xy, width, height, text, fc, ec="#333333", fontsize=9.5):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        linewidth=1.25,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1f1f1f",
    )
    return box


def add_arrow(ax, start, end, color="#555555", rad=0.0):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.4,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def save_architecture_figure():
    def node(ax, x, y, label, color, radius=0.026, fontsize=8.4):
        circ = plt.Circle((x, y), radius, facecolor=color, edgecolor="#2d2d2d", linewidth=1.1)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color="#202020")

    def small_graph(ax, x, y, label):
        coords = [(x, y), (x + 0.048, y + 0.038), (x + 0.095, y), (x + 0.046, y - 0.043)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        for i, j in edges:
            ax.plot(
                [coords[i][0], coords[j][0]],
                [coords[i][1], coords[j][1]],
                color="#6c7480",
                linewidth=1.2,
                zorder=1,
            )
        for k, (px, py) in enumerate(coords):
            node(ax, px, py, "C" if k != 1 else "N", "#f7fafc", radius=0.017, fontsize=6.8)
        ax.text(x + 0.047, y - 0.075, label, ha="center", va="top", fontsize=9.5, fontweight="bold")

    fig, ax = plt.subplots(figsize=(15.4, 7.9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0.18, 1)
    ax.axis("off")

    colors = {
        "input": "#edf2f7",
        "drug": "#d9ecff",
        "cell": "#e2f1dc",
        "kan": "#fff1cf",
        "main": "#f7d8d4",
        "alt": "#f7e7be",
        "out": "#eadff6",
        "blue": "#25639c",
        "red": "#b4473f",
        "green": "#3d7b54",
    }

    ax.text(0.5, 0.955, "HgKAN model architecture for drug synergy prediction", ha="center", va="center", fontsize=14.5, fontweight="bold")
    ax.text(0.5, 0.918, "Input molecules and cell context are encoded first; KAN is inserted into drug encoding and structured interaction messages.", ha="center", va="center", fontsize=9.2, color="#444444")

    add_box(ax, (0.035, 0.65), 0.18, 0.22, "", colors["input"], fontsize=10.2)
    ax.text(0.125, 0.853, "Drug molecular graphs", ha="center", va="center", fontsize=10.2, fontweight="bold")
    small_graph(ax, 0.075, 0.79, "Drug A")
    small_graph(ax, 0.075, 0.69, "Drug B")
    add_box(ax, (0.035, 0.36), 0.18, 0.18, "", colors["input"], fontsize=10.2)
    ax.text(0.125, 0.515, "Cell line expression", ha="center", va="center", fontsize=10.2, fontweight="bold")
    ax.text(0.125, 0.375, r"$x_c$", ha="center", va="center", fontsize=10)
    for i, h in enumerate([0.065, 0.105, 0.145, 0.185]):
        ax.add_patch(plt.Rectangle((h, 0.395), 0.026, 0.09 + 0.015 * (i % 2), facecolor="#9abf88", edgecolor="white", linewidth=0.7))

    add_box(ax, (0.27, 0.61), 0.23, 0.27, "Shared DrugKAN encoder", colors["drug"], fontsize=11)
    ax.text(0.385, 0.822, "for each GNN layer", ha="center", va="center", fontsize=8.5, color="#4a5568")
    ax.text(0.385, 0.765, r"$\bar{h}_i=\frac{1}{2}(h_i+\mathrm{mean}_{j\in N(i)}h_j)$", ha="center", va="center", fontsize=10)
    ax.text(0.385, 0.710, r"$h'_i=\mathrm{skip}(h_i)+\sum_k \phi_{o,k}(\bar{h}_{i,k})$", ha="center", va="center", fontsize=10)
    ax.text(0.385, 0.655, "global pooling -> z_A, z_B", ha="center", va="center", fontsize=9.2, color="#2b4f73")
    add_box(ax, (0.27, 0.33), 0.23, 0.18, "Cell encoder\nMLP(x_c) -> z_c", colors["cell"], fontsize=10.5)

    add_arrow(ax, (0.215, 0.76), (0.27, 0.76))
    add_arrow(ax, (0.215, 0.45), (0.27, 0.42))

    add_box(ax, (0.545, 0.58), 0.19, 0.30, "", colors["main"], fontsize=10.5)
    ax.text(0.640, 0.852, "HgKAN-Agg", ha="center", va="center", fontsize=10.8, fontweight="bold")
    ax.text(0.640, 0.822, "main performance path", ha="center", va="center", fontsize=8.8, color="#7d2c28")
    node(ax, 0.595, 0.772, "zA", "#f7fafc")
    node(ax, 0.685, 0.772, "zB", "#f7fafc")
    node(ax, 0.640, 0.700, "pAB", "#fff8df", radius=0.030)
    node(ax, 0.640, 0.630, "zc", "#ecf7e8")
    add_arrow(ax, (0.607, 0.758), (0.630, 0.718))
    add_arrow(ax, (0.673, 0.758), (0.650, 0.718))
    add_arrow(ax, (0.640, 0.672), (0.640, 0.654), color=colors["red"])
    ax.text(0.640, 0.603, r"$m=\sum \phi(p_{AB}, z_c)$", ha="center", va="center", fontsize=9)
    ax.text(0.640, 0.563, "pair-pair edges when drug is shared", ha="center", va="center", fontsize=7.8, color="#6b3a36")

    add_box(ax, (0.545, 0.25), 0.19, 0.25, "", colors["alt"], fontsize=10.5)
    ax.text(0.640, 0.475, "HgKAN-HG", ha="center", va="center", fontsize=10.8, fontweight="bold")
    ax.text(0.640, 0.448, "interpretability path", ha="center", va="center", fontsize=8.8, color="#6d5313")
    node(ax, 0.585, 0.405, "D1", "#f7fafc")
    node(ax, 0.695, 0.405, "D2", "#f7fafc")
    node(ax, 0.640, 0.315, "C", "#ecf7e8")
    node(ax, 0.640, 0.385, "e", "#fff8df", radius=0.030)
    for px, py in [(0.585, 0.405), (0.695, 0.405), (0.640, 0.315)]:
        add_arrow(ax, (px, py), (0.640, 0.385), color="#8a6a1b", rad=0.08)
    ax.text(0.640, 0.270, "node -> hyperedge -> node\nKAN edge functions are sampled", ha="center", va="center", fontsize=8.0, color="#5d4b15")

    add_arrow(ax, (0.50, 0.735), (0.545, 0.735))
    add_arrow(ax, (0.50, 0.405), (0.545, 0.405))

    add_box(ax, (0.795, 0.58), 0.16, 0.22, "Decoder", colors["out"], fontsize=11)
    ax.text(0.875, 0.715, r"$P(y=1)$", ha="center", va="center", fontsize=12)
    ax.text(0.875, 0.655, "Loewe score", ha="center", va="center", fontsize=10)
    add_arrow(ax, (0.735, 0.720), (0.795, 0.700), color=colors["red"])
    add_arrow(ax, (0.735, 0.375), (0.795, 0.635), color="#8a6a1b", rad=0.12)

    add_box(ax, (0.795, 0.29), 0.16, 0.17, "Explanations", "#f2edf8", fontsize=10.5)
    ax.text(0.875, 0.392, "KAN curves", ha="center", va="center", fontsize=8.8)
    ax.text(0.875, 0.350, "drug perturbation", ha="center", va="center", fontsize=8.8)
    ax.text(0.875, 0.308, "atom saliency", ha="center", va="center", fontsize=8.8)
    add_arrow(ax, (0.735, 0.350), (0.795, 0.365), color="#8a6a1b")

    fig.tight_layout()
    path = OUT_DIR / "model_architecture_overview.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_kan_curve_grid(
    curves_path,
    output_name,
    title,
    layer_filter=None,
    x_window=(-1.6, 1.6),
    include_title=True,
):
    df = pd.read_csv(curves_path)
    if layer_filter is not None:
        df = df[df["layer"].str.contains(layer_filter, regex=False)].copy()
    ranks = sorted(df["curve_rank"].unique())[:4]
    df = df[df["curve_rank"].isin(ranks)]
    if x_window is not None:
        x_min, x_max = x_window
        df = df[(df["x"] >= x_min) & (df["x"] <= x_max)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), sharex=True)
    colors = ["#215f9a", "#b4413c", "#3b7f4f", "#7457a6"]
    for ax, rank, color in zip(axes.ravel(), ranks, colors):
        curve = df[df["curve_rank"] == rank].sort_values("x")
        meta = curve.iloc[0]
        x = curve["x"].to_numpy()
        y = curve["y"].to_numpy()
        ax.plot(x, y, color=color, linewidth=2.1)
        ax.fill_between(x, 0, y, where=y >= 0, color="#d94d3a", alpha=0.14)
        ax.fill_between(x, 0, y, where=y < 0, color="#2d74b8", alpha=0.14)
        ax.axhline(0, color="#222222", linewidth=0.8, alpha=0.65)
        ax.axvline(0, color="#9a9a9a", linewidth=0.8, linestyle=":", alpha=0.8)
        ax.grid(color="#e8e8e8", linewidth=0.75)
        ax.set_title(
            rf"$\phi_{{{int(meta.out_channel)},{int(meta.input_channel)}}}$"
            f"  strength={float(meta.nonlinear_strength):.3f}",
            fontsize=10,
        )
        ax.tick_params(labelsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("KAN input-channel value (central window)")
    for ax in axes[:, 0]:
        ax.set_ylabel("KAN edge contribution")
    if include_title and title:
        fig.suptitle(title, fontsize=12.5)
    fig.tight_layout()
    path = OUT_DIR / output_name
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_kan_edge_curves():
    return save_kan_curve_grid(
        KAN_DIR / "kan_curves_top_channels.csv",
        "case_57954_kan_edge_functions.png",
        "Top learned KAN edge functions in the hypergraph encoder",
        layer_filter="edge_func",
    )


def save_agg_kan_curves():
    return save_kan_curve_grid(
        AGG_KAN_DIR / "kan_curves_top_channels.csv",
        "hgkan_agg_cold_drug_kan_functions.png",
        "Top learned KAN functions in the drug-pair aggregation graph",
        layer_filter="kan_graph_encoder",
    )


def save_basis_comparison():
    rows = [
        ("Fourier\nDrugKAN", 19.1304, 17.8930, "#27639b"),
        ("B-spline\nDrugKAN", 18.9037, 19.0523, "#bf5a4a"),
        ("Fourier\nHgKAN-Agg", 19.2011, 17.6763, "#3f7b55"),
    ]
    labels = [row[0] for row in rows]
    random_vals = [row[1] for row in rows]
    cold_vals = [row[2] for row in rows]
    colors = [row[3] for row in rows]
    x = range(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars_random = ax.bar(
        [i - width / 2 for i in x],
        random_vals,
        width,
        label="Random split",
        color=[c for c in colors],
        alpha=0.60,
        edgecolor="#333333",
        linewidth=0.6,
    )
    bars_cold = ax.bar(
        [i + width / 2 for i in x],
        cold_vals,
        width,
        label="Cold-drug split",
        color=colors,
        alpha=0.95,
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Test RMSE (lower is better)")
    ax.set_title("Fourier vs B-spline KAN under Loewe regression")
    ax.set_xlim(-0.55, 2.75)
    ax.set_ylim(17.2, 20.45)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    split_legend = [
        Patch(facecolor="#7f8c9a", edgecolor="#333333", alpha=0.60, label="Random split (lighter bar)"),
        Patch(facecolor="#7f8c9a", edgecolor="#333333", alpha=0.95, label="Cold-drug split (darker bar)"),
    ]
    ax.legend(handles=split_legend, frameon=False, loc="upper left")
    ax.axhline(18.9204, color="#6f6f6f", linestyle=":", linewidth=1.1, alpha=0.85)
    ax.axhline(18.3997, color="#2f2f2f", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.text(
        2.68,
        18.96,
        "MLP random\n18.92",
        fontsize=7.8,
        color="#5b5b5b",
        ha="right",
        va="bottom",
    )
    ax.text(
        2.68,
        18.34,
        "MLP cold\n18.40",
        fontsize=7.8,
        color="#333333",
        ha="right",
        va="top",
    )
    ax.text(
        0.77,
        19.58,
        "B-spline is close on random,\nbut loses cold-drug generalization.",
        fontsize=8.5,
        color="#6b2d27",
    )
    ax.text(
        2.38,
        17.76,
        "best cold-drug\nRMSE",
        fontsize=8.5,
        color="#27563a",
        ha="left",
        va="center",
    )
    for bars in (bars_random, bars_cold):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.045,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                color="#2b2b2b",
            )
    fig.tight_layout()
    path = OUT_DIR / "fourier_bspline_rmse_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_response_curve(output_name="case_57954_drug_pair_response_curve.png", include_title=True):
    df = pd.read_csv(CASE_DIR / "drug_pair_response_curve.csv")
    label_map = {
        "drug_pair_retention": "Both drugs retained",
        "drug_a_retention": "Lenalidomide retained",
        "drug_b_retention": "mitomycin C retained",
    }
    color_map = {
        "drug_pair_retention": "#215f9a",
        "drug_a_retention": "#d38b36",
        "drug_b_retention": "#b4413c",
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for name in ["drug_pair_retention", "drug_a_retention", "drug_b_retention"]:
        view = df[df["intervention"] == name].sort_values("alpha")
        ax.plot(
            view["alpha"],
            view["score"],
            marker="o",
            markersize=4,
            linewidth=2.1,
            color=color_map[name],
            label=label_map[name],
        )
    ax.axhline(0.5, color="#222222", linewidth=1.1, linestyle="--", alpha=0.72)
    ax.axvspan(0.0, 0.6, color="#eef2f7", alpha=0.65, zorder=-1)
    ax.text(0.02, 0.525, "decision threshold", fontsize=8.2, color="#333333")
    ax.text(0.31, 0.12, "rapid nonlinear rise", fontsize=8.5, color="#44546a")
    ax.set_xlabel("Retention of aligned drug representation")
    ax.set_ylabel("Synergy probability")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 1.02)
    if include_title:
        ax.set_title("Drug-pair retention response on the real case")
    ax.grid(color="#e8e8e8", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    path = OUT_DIR / output_name
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def save_kan_response_link():
    kan_path = save_kan_curve_grid(
        KAN_DIR / "kan_curves_top_channels.csv",
        "case_57954_kan_edge_functions_link_compact.png",
        "Top learned KAN edge functions in the hypergraph encoder",
        layer_filter="edge_func",
        include_title=False,
    )
    response_path = save_response_curve(
        output_name="case_57954_drug_pair_response_curve_compact.png",
        include_title=False,
    )

    fig = plt.figure(figsize=(13.4, 5.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.0])
    panels = [
        (fig.add_subplot(grid[0, 0]), "A. Hidden KAN edge functions", kan_path),
        (fig.add_subplot(grid[0, 1]), "B. Drug-pair retention response", response_path),
    ]
    for ax, title, path in panels:
        ax.imshow(Image.open(path))
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=2)
    fig.subplots_adjust(top=0.94, bottom=0.04, left=0.03, right=0.99, wspace=0.04)
    path = OUT_DIR / "case_57954_kan_response_link.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    kan_path.unlink(missing_ok=True)
    response_path.unlink(missing_ok=True)
    return path


def molecule_from_preprocessing(drug_a, drug_b, cell):
    df = pd.read_csv(ROOT / "graphnn/data/preprocessing/drugcomb_loewe_thresh_1.csv")
    row = df[
        (df["Drug1_ID"].astype(str) == drug_a)
        & (df["Drug2_ID"].astype(str) == drug_b)
        & (df["Cell_Line_ID"].astype(str) == cell)
    ].iloc[0]
    return row["Drug1"], row["Drug2"]


def save_molecule_plots():
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
    except Exception as exc:  # pragma: no cover
        print(f"RDKit is unavailable, skipping molecule plots: {exc}")
        return []

    smiles_a, smiles_b = molecule_from_preprocessing(
        "Lenalidomide", "mitomycin C", "IGROV1"
    )
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)

    sal = pd.read_csv(CASE_DIR / "atom_embedding_saliency.csv")
    top_a = sal[sal["drug_slot"] == "drug_a"].head(12)
    top_b = sal[sal["drug_slot"] == "drug_b"].head(12)

    def draw(mol, top, title, filename):
        rdDepictor.Compute2DCoords(mol)
        top = top.sort_values("saliency_grad_x_activation", ascending=False).head(12)
        atom_scores = {
            int(row.local_atom_index): float(row.saliency_grad_x_activation)
            for row in top.itertuples(index=False)
            if int(row.local_atom_index) < mol.GetNumAtoms()
        }
        if not atom_scores:
            return None
        max_score = max(atom_scores.values())
        highlight_atoms = list(atom_scores)
        highlight_bonds = []
        for bond in mol.GetBonds():
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if begin in atom_scores and end in atom_scores:
                highlight_bonds.append(bond.GetIdx())
        atom_colors = {
            idx: (0.95, 0.74 - 0.42 * (score / max_score), 0.18)
            for idx, score in atom_scores.items()
        }
        bond_colors = {
            idx: (0.95, 0.36, 0.18)
            for idx in highlight_bonds
        }
        atom_radii = {
            idx: 0.16 + 0.24 * (score / max_score)
            for idx, score in atom_scores.items()
        }

        drawer = rdMolDraw2D.MolDraw2DCairo(760, 500)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.fillHighlights = True
        opts.fixedBondLength = 34
        opts.padding = 0.08
        opts.bondLineWidth = 2.0
        opts.legendFontSize = 22
        opts.highlightBondWidthMultiplier = 10
        rdMolDraw2D.PrepareAndDrawMolecule(
            drawer,
            mol,
            legend="",
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
            highlightAtomRadii=atom_radii,
        )
        drawer.FinishDrawing()
        path = OUT_DIR / filename
        path.write_bytes(drawer.GetDrawingText())
        return path

    paths = [
        draw(mol_a, top_a, "Lenalidomide: atom saliency", "case_57954_lenalidomide_saliency.png"),
        draw(mol_b, top_b, "mitomycin C: atom saliency", "case_57954_mitomycin_c_saliency.png"),
    ]
    return [p for p in paths if p is not None]


def trim_white_margin(image, padding=22):
    img = image.convert("RGB")
    diff = ImageChops.difference(img, Image.new("RGB", img.size, "white"))
    bbox = diff.getbbox()
    if bbox is None:
        return img
    return ImageOps.expand(img.crop(bbox), border=padding, fill="white")


def save_overview(paths):
    fig = plt.figure(figsize=(14.0, 10.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    panel_specs = [
        (fig.add_subplot(grid[0, 0]), "A", "Drug perturbation", paths["modality"]),
        (fig.add_subplot(grid[0, 1]), "B", "Drug-pair retention response", paths["response"]),
        (fig.add_subplot(grid[1, 0]), "C", "Lenalidomide atom saliency", paths["mol_a"]),
        (fig.add_subplot(grid[1, 1]), "D", "mitomycin C atom saliency", paths["mol_b"]),
    ]
    for ax, panel, title, path in panel_specs:
        img = Image.open(path)
        if title.endswith("atom saliency"):
            img = trim_white_margin(img, padding=24)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"{panel}. {title}", loc="left", fontsize=12, fontweight="bold", pad=2)
        if panel == "C":
            ax.text(
                0.03,
                0.045,
                "CRBN-binding glutarimide ring:\n6/6 core atoms in top-12 saliency",
                transform=ax.transAxes,
                fontsize=8.5,
                color="#2b2b2b",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", alpha=0.88),
            )
        elif panel == "D":
            ax.text(
                0.03,
                0.045,
                "DNA-alkylating quinone ring:\n6/6 core atoms in top-10 saliency",
                transform=ax.transAxes,
                fontsize=8.5,
                color="#2b2b2b",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbbbbb", alpha=0.88),
            )

    fig.subplots_adjust(top=0.965, bottom=0.035, left=0.035, right=0.985, hspace=0.16, wspace=0.06)
    path = OUT_DIR / "case_57954_visual_explanation_overview.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main():
    save_architecture_figure()
    save_basis_comparison()
    save_modality_plot()
    modality = save_modality_plot(
        output_name="case_57954_modality_perturbation_compact.png",
        include_title=False,
    )
    gene = save_gene_plot()
    save_kan_edge_curves()
    save_response_curve()
    save_kan_response_link()
    response = save_response_curve(
        output_name="case_57954_drug_pair_response_curve_compact.png",
        include_title=False,
    )
    kan = save_kan_curve_grid(
        KAN_DIR / "kan_curves_top_channels.csv",
        "case_57954_kan_edge_functions_compact.png",
        "Top learned KAN edge functions in the hypergraph encoder",
        layer_filter="edge_func",
        include_title=False,
    )
    save_agg_kan_curves()
    mol_paths = save_molecule_plots()
    paths = {"modality": modality, "gene": gene, "kan": kan, "response": response}
    if len(mol_paths) >= 1:
        paths["mol_a"] = mol_paths[0]
    if len(mol_paths) >= 2:
        paths["mol_b"] = mol_paths[1]
    overview = save_overview(paths)
    for key in ("modality", "kan", "response"):
        compact_path = paths.get(key)
        if compact_path is not None and compact_path.name.endswith("_compact.png"):
            compact_path.unlink(missing_ok=True)
    print(f"Wrote {overview}")


if __name__ == "__main__":
    main()
