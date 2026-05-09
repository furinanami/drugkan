#!/usr/bin/env python3
"""
Build or execute the recommended thesis experiment suite.

Default mode is dry-run: it writes the exact training commands to a text file.
Use --execute only on a machine where the requested GPUs are available.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run thesis experiment suite")
    parser.add_argument("--gpus", default="0,1,2,3,4,5")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--early-stopping", type=int, default=10)
    parser.add_argument("--scenarios", default="random,cold_drug")
    parser.add_argument("--run-tag-prefix", default="thesis")
    parser.add_argument(
        "--preset",
        choices=["minimal", "full"],
        default="minimal",
        help="minimal uses GCN only; full also includes kagcn_neighbor and cold_cell.",
    )
    parser.add_argument("--output", default="graphnn/thesis_experiment_commands.txt")
    parser.add_argument("--execute", action="store_true", help="Run commands sequentially")
    return parser.parse_args()


def build_commands(args):
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    scenarios = args.scenarios
    gnn_types = "gcn"
    if args.preset == "full":
        gnn_types = "gcn,kagcn_neighbor"
        if "cold_cell" not in scenarios.split(","):
            scenarios = scenarios + ",cold_cell"

    common = [
        sys.executable,
        "graphnn/run_multi_gpu_comparison.py",
        "--gpus",
        args.gpus,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--early_stopping",
        str(args.early_stopping),
        "--scenario",
        scenarios,
        "--gnn_types",
        gnn_types,
        "--use_kan",
        "both",
        "--use_hypergraph",
        "mlp,hypergraph,kan_hypergraph,kan_aggregated",
    ]

    commands = []
    for seed in seeds:
        for task in ["classification", "regression"]:
            cmd = list(common)
            cmd.extend([
                "--task",
                task,
                "--seed",
                str(seed),
                "--run_tag",
                f"{args.run_tag_prefix}_{task}_seed{seed}",
            ])
            if task == "regression":
                cmd.extend(["--target_score", "loewe"])
            commands.append(cmd)
    return commands


def shell_join(cmd):
    return " ".join(str(part) for part in cmd)


def main():
    args = parse_args()
    commands = build_commands(args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for cmd in commands:
            handle.write(shell_join(cmd) + "\n")

    print(f"Wrote {len(commands)} commands to {os.path.abspath(output_path)}")
    if not args.execute:
        print("Dry-run only. Re-run with --execute on a GPU machine to start training.")
        return

    for idx, cmd in enumerate(commands, start=1):
        print(f"\n[{idx}/{len(commands)}] {shell_join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
