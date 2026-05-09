#!/usr/bin/env python3
"""
Collect all_results.csv files and write thesis-ready mean/std summary tables.
"""
import argparse
import os
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Collect thesis experiment results")
    parser.add_argument(
        "--experiments-root",
        default="graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1/experiments",
        help="Directory containing multi_gpu_comparison_* runs",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for summary CSV/Markdown tables")
    return parser.parse_args()


def find_results(root):
    return sorted(Path(root).glob("multi_gpu_comparison_*/all_results.csv"))


def metric_columns(df, task):
    if task == "regression":
        return [c for c in ["test_rmse", "test_mae", "test_r2", "valid_rmse", "valid_mae", "valid_r2"] if c in df.columns]
    return [c for c in ["test_aupr", "test_auc", "valid_aupr", "valid_auc"] if c in df.columns]


def add_model_label(df):
    df = df.copy()
    if "variant_group" not in df.columns:
        df["variant_group"] = None
    df["variant_group"] = df["variant_group"].fillna(df["exp_name"].str.split("_").str[0])
    if "hypergraph_mode" not in df.columns:
        df["hypergraph_mode"] = None
    inferred_mode = df["exp_name"].str.extract(r"(kan_hypergraph|kan_aggregated|hypergraph|mlp)", expand=False)
    df["hypergraph_mode"] = df["hypergraph_mode"].fillna(inferred_mode).fillna("unknown")
    for column in ["use_drug_kan", "use_hypergraph_kan", "use_kan"]:
        if column not in df.columns:
            df[column] = None
    drug_kan = df["use_drug_kan"].fillna(df["use_kan"]).fillna(
        df["exp_name"].str.contains("KAGNN|KAGCN", case=False, regex=True)
    )
    hyper_kan = df["use_hypergraph_kan"].fillna(df["use_kan"]).fillna(
        df["hypergraph_mode"].astype(str).str.contains("kan", case=False, regex=False)
    )
    df["model_label"] = (
        df["variant_group"].astype(str)
        + "__"
        + df["hypergraph_mode"].astype(str)
        + "__decoder="
        + df.get("decoder_type", "mlp").astype(str)
        + "__drugKAN="
        + drug_kan.astype(str)
        + "__hyperKAN="
        + hyper_kan.astype(str)
    )
    return df


def summarize_task(df, task):
    task_df = add_model_label(df[df["task"].fillna("classification") == task])
    if task_df.empty:
        return pd.DataFrame()

    group_cols = ["task", "scenario", "model_label", "variant_group", "hypergraph_mode"]
    if "target_score" in task_df.columns and task == "regression":
        group_cols.insert(1, "target_score")

    metrics = metric_columns(task_df, task)
    agg_map = {}
    for metric in metrics:
        agg_map[f"{metric}_mean"] = (metric, "mean")
        agg_map[f"{metric}_std"] = (metric, "std")
    agg_map["n_runs"] = ("exp_name", "count")

    summary = task_df.groupby(group_cols, dropna=False).agg(**agg_map).reset_index()
    sort_metric = "test_rmse_mean" if task == "regression" and "test_rmse_mean" in summary else None
    if sort_metric:
        return summary.sort_values(["scenario", sort_metric], ascending=[True, True])
    if "test_aupr_mean" in summary:
        return summary.sort_values(["scenario", "test_aupr_mean"], ascending=[True, False])
    return summary


def write_markdown(df, path, max_rows=80):
    def format_value(value):
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    view = df.head(max_rows)
    with open(path, "w", encoding="utf-8") as handle:
        if view.empty:
            handle.write("No results.\n")
            return
        columns = list(view.columns)
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for _, row in view.iterrows():
            handle.write("| " + " | ".join(format_value(row[col]) for col in columns) + " |\n")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result_paths = find_results(args.experiments_root)
    if not result_paths:
        raise RuntimeError(f"No all_results.csv files found under {args.experiments_root}")

    frames = []
    for path in result_paths:
        df = pd.read_csv(path)
        df["results_file"] = str(path)
        df["run_dir"] = str(path.parent)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    if "task" not in all_df.columns:
        all_df["task"] = "classification"
    all_df["task"] = all_df["task"].fillna("classification")
    if "scenario" not in all_df.columns:
        all_df["scenario"] = None
    all_df["scenario"] = all_df["scenario"].fillna(
        all_df["exp_name"].str.extract(r"(random|cold_drug|cold_cell)", expand=False)
    )

    all_df.to_csv(out_dir / "thesis_all_results_raw.csv", index=False)

    for task in sorted(all_df["task"].fillna("classification").unique()):
        summary = summarize_task(all_df, task)
        if summary.empty:
            continue
        summary.to_csv(out_dir / f"thesis_{task}_summary.csv", index=False)
        write_markdown(summary, out_dir / f"thesis_{task}_summary.md")

    print(f"Collected {len(result_paths)} result files")
    print(f"Wrote summaries to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
