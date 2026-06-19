#!/usr/bin/env python3
"""
Analyze whether drug-pair feature clusters have more coherent synergy labels.

This is an exploratory script. It does not train a neural model. It builds
label-free drug and drug-pair features from molecular graph atom-feature
histograms, clusters unique unordered drug pairs, then evaluates whether the
resulting groups have lower label entropy / stronger positive-rate separation
than random label permutations. It also tests a simple group-prior predictor.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "graphnn"))

from deepadr.cold_split import get_split_by_scenario  # noqa: E402
from deepadr.dataset import MoleculeDataset  # noqa: E402
from ogb.utils.features import get_atom_feature_dims  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        default="graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1",
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--k_values", default="8,16,32,64,128")
    parser.add_argument("--random_repeats", type=int, default=100)
    parser.add_argument("--min_cluster_size", type=int, default=20)
    parser.add_argument("--scenarios", default="random,cold_drug")
    return parser.parse_args()


def atom_hist_feature(x, dims):
    cols = []
    x = x.long()
    denom = max(int(x.shape[0]), 1)
    for j, dim in enumerate(dims):
        vals = x[:, j].clamp(min=0, max=dim - 1)
        hist = torch.bincount(vals, minlength=dim).float() / denom
        cols.append(hist)
    cols.append(torch.tensor([math.log1p(denom)], dtype=torch.float32))
    return torch.cat(cols).numpy()


def build_drug_features(dataset):
    data = dataset.data
    slices = dataset.slices
    dims = get_atom_feature_dims()

    drug_features = {}
    drug_a = data.drug_a_id.cpu().numpy()
    drug_b = data.drug_b_id.cpu().numpy()
    n = len(drug_a)

    for i in range(n):
        da = int(drug_a[i])
        if da not in drug_features:
            lo = int(slices["x_a"][i])
            hi = int(slices["x_a"][i + 1])
            drug_features[da] = atom_hist_feature(data.x_a[lo:hi].cpu(), dims)

        db = int(drug_b[i])
        if db not in drug_features:
            lo = int(slices["x_b"][i])
            hi = int(slices["x_b"][i + 1])
            drug_features[db] = atom_hist_feature(data.x_b[lo:hi].cpu(), dims)

    return drug_features


def build_pair_table(dataset, drug_features):
    data = dataset.data
    drug_a = data.drug_a_id.cpu().numpy().astype(int)
    drug_b = data.drug_b_id.cpu().numpy().astype(int)
    cell = data.cell_id.cpu().numpy().astype(int)
    y = data.y.cpu().numpy().astype(int)
    loewe = data.loewe_score.cpu().numpy() if hasattr(data, "loewe_score") else None

    pair_to_idx = {}
    pair_keys = []
    sample_pair_idx = np.empty(len(y), dtype=np.int64)

    for i, (a, b) in enumerate(zip(drug_a, drug_b)):
        key = tuple(sorted((int(a), int(b))))
        idx = pair_to_idx.get(key)
        if idx is None:
            idx = len(pair_keys)
            pair_to_idx[key] = idx
            pair_keys.append(key)
        sample_pair_idx[i] = idx

    pair_features = []
    for a, b in pair_keys:
        fa = drug_features[a]
        fb = drug_features[b]
        pair_features.append(np.concatenate([
            (fa + fb) * 0.5,
            np.abs(fa - fb),
            fa * fb,
        ]))
    pair_features = np.asarray(pair_features, dtype=np.float32)

    samples = pd.DataFrame({
        "sample_idx": np.arange(len(y), dtype=np.int64),
        "pair_idx": sample_pair_idx,
        "drug_a": drug_a,
        "drug_b": drug_b,
        "cell": cell,
        "y": y,
    })
    if loewe is not None:
        samples["loewe_score"] = loewe

    pairs = pd.DataFrame(pair_keys, columns=["drug_1", "drug_2"])
    pairs.insert(0, "pair_idx", np.arange(len(pair_keys), dtype=np.int64))

    return samples, pairs, pair_features


def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def cluster_metrics(y, cluster, min_cluster_size=20):
    df = pd.DataFrame({"y": y.astype(float), "cluster": cluster.astype(int)})
    g = df.groupby("cluster")["y"].agg(["count", "mean"])
    g = g[g["count"] >= min_cluster_size]
    if g.empty:
        return {
            "n_clusters_used": 0,
            "weighted_entropy": np.nan,
            "purity": np.nan,
            "pos_rate_std": np.nan,
            "mean_abs_deviation": np.nan,
            "top_cluster_pos_rate": np.nan,
            "bottom_cluster_pos_rate": np.nan,
        }
    weights = g["count"].to_numpy() / g["count"].sum()
    rates = g["mean"].to_numpy()
    global_rate = float(df["y"].mean())
    return {
        "n_clusters_used": int(len(g)),
        "weighted_entropy": float(np.sum(weights * binary_entropy(rates))),
        "purity": float(np.sum(weights * np.maximum(rates, 1 - rates))),
        "pos_rate_std": float(np.sqrt(np.average((rates - global_rate) ** 2, weights=weights))),
        "mean_abs_deviation": float(np.sum(weights * np.abs(rates - global_rate))),
        "top_cluster_pos_rate": float(np.max(rates)),
        "bottom_cluster_pos_rate": float(np.min(rates)),
    }


def random_label_baseline(y, cluster, repeats, rng, min_cluster_size):
    rows = []
    y = np.asarray(y)
    for _ in range(repeats):
        rows.append(cluster_metrics(rng.permutation(y), cluster, min_cluster_size))
    return pd.DataFrame(rows).agg(["mean", "std"]).T


def group_prior_predict(train_y, train_cluster, eval_cluster, smoothing=5.0):
    global_rate = float(np.mean(train_y))
    stat = pd.DataFrame({"y": train_y, "cluster": train_cluster}).groupby("cluster")["y"].agg(["sum", "count"])
    prior = {
        int(k): float((row["sum"] + smoothing * global_rate) / (row["count"] + smoothing))
        for k, row in stat.iterrows()
    }
    return np.asarray([prior.get(int(c), global_rate) for c in eval_cluster], dtype=float)


def score_predictions(y, score):
    y = np.asarray(y).astype(int)
    out = {"aupr": average_precision_score(y, score)}
    if len(np.unique(y)) == 2:
        out["auc"] = roc_auc_score(y, score)
    else:
        out["auc"] = np.nan
    return out


def main():
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    out_dir = Path(args.out_dir or Path(args.dataset_root) / "experiments" / "pair_cluster_modes")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset_root}")
    dataset = MoleculeDataset(root=args.dataset_root, dataset="tdcSynergy")
    print(f"Samples: {len(dataset)}")

    print("Building label-free drug atom-histogram features...")
    drug_features = build_drug_features(dataset)
    print(f"Unique drugs: {len(drug_features)}")

    print("Building unordered pair table and pair features...")
    samples, pairs, pair_features = build_pair_table(dataset, drug_features)
    print(f"Unique pairs: {len(pairs)}")

    scaler = StandardScaler()
    pair_features_z = scaler.fit_transform(pair_features)

    samples.to_csv(out_dir / "sample_pair_table.csv", index=False)
    pairs.to_csv(out_dir / "unique_pairs.csv", index=False)

    k_values = [int(x) for x in args.k_values.split(",") if x.strip()]
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    summary_rows = []
    cluster_assignments = {"pair_idx": pairs["pair_idx"].to_numpy()}

    y_all = samples["y"].to_numpy().astype(int)
    sample_pair_idx = samples["pair_idx"].to_numpy()

    for k in k_values:
        print(f"Clustering unique pairs: k={k}")
        km = MiniBatchKMeans(
            n_clusters=k,
            random_state=args.seed,
            batch_size=4096,
            n_init=10,
            max_iter=300,
        )
        pair_cluster = km.fit_predict(pair_features_z)
        cluster_assignments[f"k{k}"] = pair_cluster
        sample_cluster = pair_cluster[sample_pair_idx]

        metrics = cluster_metrics(y_all, sample_cluster, args.min_cluster_size)
        rand = random_label_baseline(
            y_all, sample_cluster, args.random_repeats, rng, args.min_cluster_size
        )
        row = {
            "scope": "all",
            "scenario": "all",
            "k": k,
            "n_samples": len(y_all),
            "global_pos_rate": float(y_all.mean()),
            **metrics,
        }
        for key in metrics:
            if key in rand.index:
                row[f"random_{key}_mean"] = float(rand.loc[key, "mean"])
                row[f"random_{key}_std"] = float(rand.loc[key, "std"])
        summary_rows.append(row)

        for scenario in scenarios:
            part = get_split_by_scenario(
                dataset, scenario=scenario, fold=args.fold, n_folds=5, seed=args.seed
            )
            if "valid" in part and "validation" not in part:
                part["validation"] = part["valid"]
            for split_name in ["train", "validation", "test"]:
                idx = np.asarray(part[split_name], dtype=int)
                yy = y_all[idx]
                cc = sample_cluster[idx]
                metrics = cluster_metrics(yy, cc, args.min_cluster_size)
                row = {
                    "scope": split_name,
                    "scenario": scenario,
                    "k": k,
                    "n_samples": len(idx),
                    "global_pos_rate": float(yy.mean()),
                    **metrics,
                }
                summary_rows.append(row)

            train_idx = np.asarray(part["train"], dtype=int)
            test_idx = np.asarray(part["test"], dtype=int)
            valid_idx = np.asarray(part["validation"], dtype=int)
            for split_name, idx in [("validation", valid_idx), ("test", test_idx)]:
                pred = group_prior_predict(
                    y_all[train_idx],
                    sample_cluster[train_idx],
                    sample_cluster[idx],
                )
                scores = score_predictions(y_all[idx], pred)
                summary_rows.append({
                    "scope": f"{split_name}_group_prior",
                    "scenario": scenario,
                    "k": k,
                    "n_samples": len(idx),
                    "global_pos_rate": float(y_all[idx].mean()),
                    "group_prior_aupr": float(scores["aupr"]),
                    "group_prior_auc": float(scores["auc"]),
                })

    cluster_df = pd.DataFrame(cluster_assignments)
    summary = pd.DataFrame(summary_rows)
    cluster_df.to_csv(out_dir / "pair_cluster_assignments.csv", index=False)
    summary.to_csv(out_dir / "pair_cluster_summary.csv", index=False)

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\nBest group-prior test rows by scenario:")
    gp = summary[summary["scope"].eq("test_group_prior")].copy()
    if not gp.empty:
        print(gp.sort_values(["scenario", "group_prior_aupr"], ascending=[True, False])
              [["scenario", "k", "group_prior_aupr", "group_prior_auc"]]
              .groupby("scenario").head(3).to_string(index=False))

    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
