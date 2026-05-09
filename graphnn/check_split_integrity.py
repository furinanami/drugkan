#!/usr/bin/env python3
"""
Check split sizes and leakage for random/cold-drug/cold-cell thesis experiments.
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from deepadr.dataset import MoleculeDataset  # noqa: E402
from deepadr.cold_split import get_split_by_scenario  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Check DrugComb split integrity")
    parser.add_argument(
        "--dataset-root",
        default="graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1",
        help="MoleculeDataset root",
    )
    parser.add_argument("--scenarios", default="random,cold_drug,cold_cell")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    parser.add_argument(
        "--prefer-saved",
        action="store_true",
        help="Use partition_<scenario>.pkl when seed=42 and fold=0.",
    )
    return parser.parse_args()


def normalize_partition(partition):
    partition = dict(partition)
    if "valid" in partition and "validation" not in partition:
        partition["validation"] = partition["valid"]
    return partition


def load_partition(dataset, dataset_root, scenario, seed, fold, prefer_saved):
    partition_path = Path(dataset_root) / "partitions" / f"partition_{scenario}.pkl"
    if prefer_saved and seed == 42 and fold == 0 and partition_path.exists():
        with open(partition_path, "rb") as handle:
            return normalize_partition(pickle.load(handle)), str(partition_path)
    return normalize_partition(
        get_split_by_scenario(dataset, scenario=scenario, fold=fold, n_folds=5, seed=seed)
    ), "generated"


def ids_for(indices, values):
    return set(int(values[i]) for i in indices)


def pair_ids_for(indices, drug_a, drug_b):
    return set(tuple(sorted((int(drug_a[i]), int(drug_b[i])))) for i in indices)


def heldout_drugs_for(partition, split_name):
    keys = {
        "validation": ("heldout_validation_drugs", "validation_drugs", "valid_drugs"),
        "test": ("heldout_test_drugs", "test_drugs"),
    }[split_name]
    for key in keys:
        if key in partition:
            return set(int(x) for x in partition[key])
    return None


def main():
    args = parse_args()
    dataset = MoleculeDataset(root=args.dataset_root, dataset="tdcSynergy")
    drug_a = dataset.data.drug_a_id.cpu().numpy()
    drug_b = dataset.data.drug_b_id.cpu().numpy()
    cell = dataset.data.cell_id.cpu().numpy()
    y = dataset.data.y.cpu().numpy()

    rows = []
    for scenario in [x.strip() for x in args.scenarios.split(",") if x.strip()]:
        for seed in [int(x.strip()) for x in args.seeds.split(",") if x.strip()]:
            partition, source = load_partition(dataset, args.dataset_root, scenario, seed, args.fold, args.prefer_saved)
            train = partition["train"]
            valid = partition["validation"]
            test = partition["test"]

            train_drugs = ids_for(train, drug_a) | ids_for(train, drug_b)
            test_drugs = ids_for(test, drug_a) | ids_for(test, drug_b)
            train_cells = ids_for(train, cell)
            test_cells = ids_for(test, cell)
            train_pairs = pair_ids_for(train, drug_a, drug_b)
            test_pairs = pair_ids_for(test, drug_a, drug_b)
            heldout_test_drugs = heldout_drugs_for(partition, "test")
            if heldout_test_drugs is None:
                heldout_test_drugs = test_drugs if scenario == "cold_drug" else set()
            heldout_validation_drugs = heldout_drugs_for(partition, "validation") or set()
            heldout_test_train_overlap = len(train_drugs & heldout_test_drugs)
            heldout_validation_train_overlap = len(train_drugs & heldout_validation_drugs)

            rows.append(
                {
                    "scenario": scenario,
                    "seed": seed,
                    "fold": args.fold,
                    "source": source,
                    "n_train": len(train),
                    "n_validation": len(valid),
                    "n_test": len(test),
                    "train_pos_rate": float(y[train].mean()) if train else None,
                    "validation_pos_rate": float(y[valid].mean()) if valid else None,
                    "test_pos_rate": float(y[test].mean()) if test else None,
                    "drug_overlap_train_test": len(train_drugs & test_drugs),
                    "heldout_test_drugs": len(heldout_test_drugs),
                    "heldout_validation_drugs": len(heldout_validation_drugs),
                    "heldout_test_drug_train_overlap": heldout_test_train_overlap,
                    "heldout_validation_drug_train_overlap": heldout_validation_train_overlap,
                    "cell_overlap_train_test": len(train_cells & test_cells),
                    "pair_overlap_train_test": len(train_pairs & test_pairs),
                    "passes_cold_drug": scenario != "cold_drug" or heldout_test_train_overlap == 0,
                    "passes_cold_cell": scenario != "cold_cell" or len(train_cells & test_cells) == 0,
                }
            )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nWrote {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
