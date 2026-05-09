#!/usr/bin/env python3
"""
Export KAN function curves and channel importances from a trained checkpoint.

This script is intentionally checkpoint-only: it does not need the dataset or a
model class. It scans state_dict tensors for FourierKAN/BSplineKAN parameters
and writes CSV summaries plus plots for the strongest input-output channel
functions.
"""
import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Explain trained KAN layers")
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt or a state_dict")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV and PNG outputs")
    parser.add_argument(
        "--state-dict-key",
        default="auto",
        help="Checkpoint key to inspect, e.g. model_state_dict or gnn_model_state_dict. Default: auto",
    )
    parser.add_argument("--top-k", type=int, default=24, help="Number of strongest channel curves to plot")
    parser.add_argument("--num-points", type=int, default=301, help="Points per plotted curve")
    parser.add_argument("--x-min", type=float, default=-3.0, help="Raw input sweep minimum")
    parser.add_argument("--x-max", type=float, default=3.0, help="Raw input sweep maximum")
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV files")
    return parser.parse_args()


def is_tensor_dict(obj):
    return isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values())


def iter_state_dicts(checkpoint, state_dict_key):
    if state_dict_key != "auto":
        if state_dict_key not in checkpoint:
            raise KeyError(f"Missing state_dict key: {state_dict_key}")
        yield state_dict_key, checkpoint[state_dict_key]
        return

    if is_tensor_dict(checkpoint):
        yield "state_dict", checkpoint
        return

    known_keys = [
        "model_state_dict",
        "gnn_model_state_dict",
        "hypergraph_model_state_dict",
        "state_dict",
    ]
    yielded = False
    for key in known_keys:
        value = checkpoint.get(key) if isinstance(checkpoint, dict) else None
        if is_tensor_dict(value):
            yielded = True
            yield key, value

    if not yielded and isinstance(checkpoint, dict):
        tensor_items = {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}
        if tensor_items:
            yield "state_dict", tensor_items


def load_kan_layers(checkpoint_path, state_dict_key):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    layers = []

    for dict_name, state_dict in iter_state_dicts(checkpoint, state_dict_key):
        for key, coeffs in state_dict.items():
            if key.endswith(".fourier_coeffs"):
                base = key[: -len(".fourier_coeffs")]
                layers.append(
                    {
                        "state_dict": dict_name,
                        "name": f"{dict_name}.{base}",
                        "type": "fourier",
                        "coeffs": coeffs.detach().float().cpu(),
                    }
                )
            elif key.endswith(".spline_coeffs"):
                base = key[: -len(".spline_coeffs")]
                knots = state_dict.get(f"{base}.knots")
                layers.append(
                    {
                        "state_dict": dict_name,
                        "name": f"{dict_name}.{base}",
                        "type": "bspline",
                        "coeffs": coeffs.detach().float().cpu(),
                        "knots": None if knots is None else knots.detach().float().cpu(),
                    }
                )
    return layers


def fourier_channel_importance(layer):
    coeffs = layer["coeffs"]
    out_dim, in_dim, n_coeff = coeffs.shape
    n_freq = (n_coeff - 1) // 2
    rows = []
    freq_weights = torch.arange(1, n_freq + 1, dtype=coeffs.dtype)

    for out_channel in range(out_dim):
        for in_channel in range(in_dim):
            c = coeffs[out_channel, in_channel]
            cos_c = c[1 : 1 + n_freq]
            sin_c = c[1 + n_freq :]
            harmonic = torch.cat([cos_c, sin_c])
            roughness = torch.sqrt(
                ((freq_weights * cos_c) ** 2).sum() + ((freq_weights * sin_c) ** 2).sum()
            )
            rows.append(
                {
                    "layer": layer["name"],
                    "kan_type": "fourier",
                    "out_channel": out_channel,
                    "input_channel": in_channel,
                    "total_l2": float(torch.linalg.vector_norm(c)),
                    "harmonic_l2": float(torch.linalg.vector_norm(harmonic)),
                    "bias_abs": float(c[0].abs()),
                    "roughness": float(roughness),
                    "max_abs": float(c.abs().max()),
                }
            )
    return rows


def bspline_channel_importance(layer):
    coeffs = layer["coeffs"]
    out_dim, in_dim, _ = coeffs.shape
    rows = []
    for out_channel in range(out_dim):
        for in_channel in range(in_dim):
            c = coeffs[out_channel, in_channel]
            diffs = torch.diff(c)
            rows.append(
                {
                    "layer": layer["name"],
                    "kan_type": "bspline",
                    "out_channel": out_channel,
                    "input_channel": in_channel,
                    "total_l2": float(torch.linalg.vector_norm(c)),
                    "harmonic_l2": float(torch.linalg.vector_norm(diffs)),
                    "bias_abs": float(c.mean().abs()),
                    "roughness": float(torch.linalg.vector_norm(torch.diff(c, n=2)) if c.numel() > 2 else 0.0),
                    "max_abs": float(c.abs().max()),
                }
            )
    return rows


def evaluate_fourier(coeffs, out_channel, input_channel, x_values):
    c = coeffs[out_channel, input_channel]
    n_freq = (c.numel() - 1) // 2
    frequencies = torch.arange(1, n_freq + 1, dtype=x_values.dtype)
    angles = x_values[:, None] * frequencies[None, :] * math.pi
    y = c[0] + torch.cos(angles).matmul(c[1 : 1 + n_freq]) + torch.sin(angles).matmul(c[1 + n_freq :])
    return y


def bspline_basis(x, knots, i, degree):
    if degree == 0:
        if i == len(knots) - 2:
            return ((x >= knots[i]) & (x <= knots[i + 1])).float()
        return ((x >= knots[i]) & (x < knots[i + 1])).float()

    denom1 = knots[i + degree] - knots[i]
    denom2 = knots[i + degree + 1] - knots[i + 1]
    term1 = torch.zeros_like(x)
    term2 = torch.zeros_like(x)
    if abs(float(denom1)) > 1e-6:
        term1 = ((x - knots[i]) / denom1) * bspline_basis(x, knots, i, degree - 1)
    if abs(float(denom2)) > 1e-6:
        term2 = ((knots[i + degree + 1] - x) / denom2) * bspline_basis(x, knots, i + 1, degree - 1)
    return term1 + term2


def evaluate_bspline(coeffs, knots, out_channel, input_channel, x_values):
    c = coeffs[out_channel, input_channel]
    if knots is None:
        num_knots = c.numel() - 2
        degree = 3
        base_knots = torch.linspace(-1, 1, num_knots, dtype=x_values.dtype)
        knots = torch.cat([base_knots[0].repeat(degree), base_knots, base_knots[-1].repeat(degree)])
    else:
        knots = knots.to(dtype=x_values.dtype)
        degree = int(len(knots) - c.numel() - 1)
    x_norm = torch.tanh(x_values)
    basis = torch.stack([bspline_basis(x_norm, knots, i, degree) for i in range(c.numel())], dim=1)
    return basis.matmul(c)


def write_layer_summaries(importance_df, out_dir):
    layer_df = (
        importance_df.groupby(["layer", "kan_type"], as_index=False)
        .agg(
            n_edges=("total_l2", "size"),
            mean_total_l2=("total_l2", "mean"),
            max_total_l2=("total_l2", "max"),
            mean_nonlinear_strength=("harmonic_l2", "mean"),
            max_nonlinear_strength=("harmonic_l2", "max"),
            mean_roughness=("roughness", "mean"),
            max_roughness=("roughness", "max"),
        )
        .sort_values("max_nonlinear_strength", ascending=False)
    )
    layer_df.to_csv(out_dir / "kan_layer_summary.csv", index=False)

    input_df = (
        importance_df.groupby(["layer", "kan_type", "input_channel"], as_index=False)
        .agg(
            mean_total_l2=("total_l2", "mean"),
            max_total_l2=("total_l2", "max"),
            mean_nonlinear_strength=("harmonic_l2", "mean"),
            max_nonlinear_strength=("harmonic_l2", "max"),
            mean_roughness=("roughness", "mean"),
        )
        .sort_values("max_nonlinear_strength", ascending=False)
    )
    input_df.to_csv(out_dir / "kan_input_channel_summary.csv", index=False)


def make_plots(curves_df, out_dir):
    import matplotlib.pyplot as plt

    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    for rank, (_, curve_df) in enumerate(curves_df.groupby("curve_rank"), start=1):
        first = curve_df.iloc[0]
        plt.figure(figsize=(5.5, 3.5))
        plt.plot(curve_df["x"], curve_df["y"], linewidth=2)
        plt.axhline(0, color="black", linewidth=0.7, alpha=0.4)
        plt.xlabel("input channel value")
        plt.ylabel("pre-normalization contribution")
        plt.title(
            f"rank {rank}: in {int(first.input_channel)} -> out {int(first.out_channel)}\n"
            f"{first.layer}"
        )
        plt.tight_layout()
        plt.savefig(curves_dir / f"rank_{rank:03d}_curve.png", dpi=150)
        plt.close()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = load_kan_layers(args.checkpoint, args.state_dict_key)
    if not layers:
        raise RuntimeError("No FourierKAN or BSplineKAN parameters were found in the checkpoint.")

    importance_rows = []
    for layer in layers:
        if layer["type"] == "fourier":
            importance_rows.extend(fourier_channel_importance(layer))
        else:
            importance_rows.extend(bspline_channel_importance(layer))

    importance_df = pd.DataFrame(importance_rows).sort_values("harmonic_l2", ascending=False)
    importance_df.to_csv(out_dir / "kan_channel_importance.csv", index=False)
    write_layer_summaries(importance_df, out_dir)

    x_values = torch.linspace(args.x_min, args.x_max, args.num_points)
    top_df = importance_df.head(args.top_k).reset_index(drop=True)
    layer_by_name = {layer["name"]: layer for layer in layers}
    curve_rows = []
    for rank, row in top_df.iterrows():
        layer = layer_by_name[row["layer"]]
        out_channel = int(row["out_channel"])
        input_channel = int(row["input_channel"])
        if layer["type"] == "fourier":
            y_values = evaluate_fourier(layer["coeffs"], out_channel, input_channel, x_values)
        else:
            y_values = evaluate_bspline(layer["coeffs"], layer.get("knots"), out_channel, input_channel, x_values)
        for x, y in zip(x_values.tolist(), y_values.tolist()):
            curve_rows.append(
                {
                    "curve_rank": rank + 1,
                    "layer": layer["name"],
                    "kan_type": layer["type"],
                    "out_channel": out_channel,
                    "input_channel": input_channel,
                    "x": x,
                    "y": y,
                    "nonlinear_strength": float(row["harmonic_l2"]),
                }
            )

    curves_df = pd.DataFrame(curve_rows)
    curves_df.to_csv(out_dir / "kan_curves_top_channels.csv", index=False)
    if not args.no_plots:
        make_plots(curves_df, out_dir)

    print(f"Found {len(layers)} KAN layers")
    print(f"Wrote outputs to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
