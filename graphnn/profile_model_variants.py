#!/usr/bin/env python3
"""
Profile parameter counts for thesis model variants.

This is intentionally lightweight: it instantiates models from hyperparameter
files or named defaults and reports total/trainable parameters plus KAN
parameter counts.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from deepadr.model_hypergraph import DeepDDS_Hypergraph  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Profile model variants")
    parser.add_argument("--output", default="graphnn/model_profile_summary.csv")
    parser.add_argument("--hyperparameters", nargs="*", default=[], help="Optional hyperparameters.json files")
    return parser.parse_args()


def default_hp():
    return {
        "emb_dim": 128,
        "expression_input_size": 946,
        "exp_H1": 8192,
        "exp_H2": 4096,
        "unified_dim": 128,
        "hypergraph_hidden": 256,
        "decoder_hidden1": 256,
        "decoder_hidden2": 128,
        "num_attn_heads": 10,
        "p_dropout": 0.2,
        "gnn_type": "gcn",
        "num_layer": 5,
        "drug_jk": "last",
        "use_drug_readout_kan": False,
        "graph_pooling": "mean",
        "task": "classification",
        "kan_type": "fourier",
    }


def make_kwargs(hp):
    task = hp.get("task", "classification")
    return {
        "num_features_xd": 9,
        "gat_output_dim": hp.get("emb_dim", 128),
        "expression_input_size": hp.get("expression_input_size", 946),
        "exp_H1": hp.get("exp_H1", 8192),
        "exp_H2": hp.get("exp_H2", 4096),
        "unified_dim": hp.get("unified_dim", 128),
        "hypergraph_hidden": hp.get("hypergraph_hidden", 256),
        "decoder_hidden1": hp.get("decoder_hidden1", 256),
        "decoder_hidden2": hp.get("decoder_hidden2", 128),
        "num_classes": 1 if task == "regression" else 2,
        "dropout": hp.get("p_dropout", 0.2),
        "num_attn_heads": hp.get("num_attn_heads", 10),
        "hypergraph_mode": hp.get("hypergraph_mode", "mlp"),
        "use_kan": hp.get("use_hypergraph_kan", hp.get("use_kan", False)),
        "kan_type": hp.get("kan_type", "fourier"),
        "use_drug_kan": hp.get("use_drug_kan", hp.get("use_kan", False)),
        "use_drug_readout_kan": hp.get("use_drug_readout_kan", False),
        "gnn_type": hp.get("gnn_type", "gcn"),
        "num_layer": hp.get("num_layer", 5),
        "drug_jk": hp.get("drug_jk", "last"),
        "graph_pooling": hp.get("graph_pooling", "mean"),
        "decoder_type": hp.get("decoder_type", "kan" if hp.get("hypergraph_mode") == "kan_mlp" else "mlp"),
        "task": task,
    }


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    kan = sum(p.numel() for name, p in model.named_parameters() if "kan" in name or "fourier_coeffs" in name or "spline_coeffs" in name)
    decoder = sum(p.numel() for name, p in model.named_parameters() if name.startswith("decoder."))
    return total, trainable, kan, decoder


def named_variants():
    variants = []
    for mode in ["mlp", "kan_mlp", "hypergraph", "kan_hypergraph", "kan_aggregated"]:
        for drug_kan in [False, True]:
            hp = default_hp()
            hp.update({
                "name": f"{'KAGCN' if drug_kan else 'GCN'}_{mode}",
                "hypergraph_mode": mode,
                "use_kan": drug_kan or mode.startswith("kan"),
                "use_drug_kan": drug_kan,
                "use_hypergraph_kan": mode in ["kan_hypergraph", "kan_aggregated"],
                "decoder_type": "kan" if mode == "kan_mlp" else "mlp",
            })
            variants.append(hp)
    return variants


def extra_variants():
    hp = default_hp()
    hp.update({
        "name": "KAGCN_multilayer_kan_mlp_drugreadoutkan",
        "hypergraph_mode": "kan_mlp",
        "use_kan": True,
        "use_drug_kan": True,
        "use_hypergraph_kan": False,
        "use_drug_readout_kan": True,
        "drug_jk": "concat_kan",
        "decoder_type": "kan",
    })
    return [hp]


def main():
    args = parse_args()
    hps = named_variants() + extra_variants()
    for hp_path in args.hyperparameters:
        with open(hp_path, "r", encoding="utf-8") as handle:
            hp = json.load(handle)
        hp["name"] = Path(hp_path).parent.name
        hps.append(hp)

    rows = []
    for hp in hps:
        model = DeepDDS_Hypergraph(**make_kwargs(hp))
        total, trainable, kan, decoder = count_params(model)
        rows.append({
            "name": hp.get("name", "variant"),
            "task": hp.get("task", "classification"),
            "hypergraph_mode": hp.get("hypergraph_mode", "mlp"),
            "decoder_type": hp.get("decoder_type", "mlp"),
            "use_drug_kan": hp.get("use_drug_kan", hp.get("use_kan", False)),
            "use_drug_readout_kan": hp.get("use_drug_readout_kan", False),
            "drug_jk": hp.get("drug_jk", "last"),
            "use_hypergraph_kan": hp.get("use_hypergraph_kan", hp.get("use_kan", False)),
            "kan_type": hp.get("kan_type", "fourier"),
            "total_params": total,
            "trainable_params": trainable,
            "kan_params": kan,
            "decoder_params": decoder,
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
