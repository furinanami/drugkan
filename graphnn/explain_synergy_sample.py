#!/usr/bin/env python3
"""
Single-sample explanations for DeepDDS_Hypergraph checkpoints.

Outputs:
  - modality_perturbation.csv: drug A, drug B, and cell embedding ablations.
  - expression_gradient_importance.csv: gene/expression feature gradient scores.
  - atom_embedding_saliency.csv: atom-level saliency on the initial atom
    embeddings for drug A and drug B.
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import torch
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from deepadr.dataset import MoleculeDataset  # noqa: E402
from deepadr.model_hypergraph import DeepDDS_Hypergraph  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Explain one drug synergy sample")
    parser.add_argument("--exp-dir", required=True, help="Experiment directory containing hyperparameters.json")
    parser.add_argument(
        "--dataset-root",
        default="graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1",
        help="MoleculeDataset root",
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Default: exp-dir/modelstates/best_model.pt")
    parser.add_argument("--sample-index", type=int, required=True, help="Dataset index to explain")
    parser.add_argument("--output-dir", default=None, help="Output directory. Default: exp-dir/explanations/sample_<idx>")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N")
    parser.add_argument("--target-class", type=int, default=1, help="Class log-probability to explain")
    parser.add_argument("--top-k", type=int, default=100, help="Rows to keep in top importance CSVs")
    parser.add_argument("--gene-names", default=None, help="Optional newline or CSV file with expression feature names")
    return parser.parse_args()


def load_gene_names(path):
    if path is None:
        return None
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if "gene" in df.columns:
            return df["gene"].astype(str).tolist()
        return df.iloc[:, 0].astype(str).tolist()
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def model_kwargs_from_hp(hp):
    task = hp.get("task", "classification")
    num_classes = 1 if task == "regression" else 2
    return {
        "num_features_xd": 9,
        "gat_output_dim": hp["emb_dim"],
        "expression_input_size": hp["expression_input_size"],
        "exp_H1": hp["exp_H1"],
        "exp_H2": hp["exp_H2"],
        "unified_dim": hp.get("unified_dim", 128),
        "hypergraph_hidden": hp.get("hypergraph_hidden", 256),
        "decoder_hidden1": hp.get("decoder_hidden1", 256),
        "decoder_hidden2": hp.get("decoder_hidden2", 128),
        "num_classes": num_classes,
        "dropout": hp["p_dropout"],
        "num_attn_heads": hp["num_attn_heads"],
        "hypergraph_mode": hp.get("hypergraph_mode", "hypergraph"),
        "use_kan": hp.get("use_hypergraph_kan", hp.get("use_kan", True)),
        "kan_type": hp.get("kan_type", "fourier"),
        "use_drug_kan": hp.get("use_drug_kan", hp.get("use_kan", True)),
        "use_drug_readout_kan": hp.get("use_drug_readout_kan", False),
        "gnn_type": hp.get("gnn_type", "gcn"),
        "num_layer": hp.get("num_layer", 5),
        "drug_jk": hp.get("drug_jk", "last"),
        "graph_pooling": hp.get("graph_pooling", "mean"),
        "decoder_type": hp.get("decoder_type", "kan" if hp.get("hypergraph_mode") == "kan_mlp" else "mlp"),
        "task": task,
    }


def load_model(exp_dir, checkpoint_path, device):
    hp_path = Path(exp_dir) / "hyperparameters.json"
    with open(hp_path, "r", encoding="utf-8") as handle:
        hp = json.load(handle)

    model = DeepDDS_Hypergraph(**model_kwargs_from_hp(hp)).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise RuntimeError(
            "Checkpoint must contain model_state_dict. Re-run hypergraph training after the checkpoint save patch."
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys while loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys while loading checkpoint: {len(unexpected)}")
    model.eval()
    return model, hp


def predict_from_aligned(model, batch, h_a_aligned, h_b_aligned, h_e_aligned):
    if model.hypergraph_mode in ["hypergraph", "kan_hypergraph"]:
        node_features, hyperedge_index, _, batch_node_indices = model.hypergraph_builder.build_hypergraph(
            batch, h_a_aligned, h_b_aligned, h_e_aligned
        )
        node_embeddings = model.hypergraph_encoder(node_features, hyperedge_index)
        h_a = node_embeddings[batch_node_indices["drug_a"]]
        h_b = node_embeddings[batch_node_indices["drug_b"]]
        h_e = node_embeddings[batch_node_indices["cell"]]
    elif model.hypergraph_mode == "kan_aggregated":
        node_features, edge_index, _, batch_node_indices = model.kan_graph_builder.build_graph(
            batch, h_a_aligned, h_b_aligned, h_e_aligned
        )
        node_embeddings = model.kan_graph_encoder(node_features, edge_index)
        drug_pair = node_embeddings[batch_node_indices["drug_pair"]]
        h_a = drug_pair
        h_b = drug_pair
        h_e = node_embeddings[batch_node_indices["cell"]]
    else:
        h_a = h_a_aligned
        h_b = h_b_aligned
        h_e = h_e_aligned
    return model.decoder(h_a, h_b, h_e)


def selected_score(output, hp, target_class):
    if hp.get("task", "classification") == "regression":
        score = output.view(-1)[0]
        if hp.get("standardize_regression_target", True):
            mean = float(hp.get("target_score_mean", 0.0))
            std = float(hp.get("target_score_std", 1.0))
            score = score * std + mean
        return score
    return output[0, target_class]


def display_score(output, hp, target_class):
    if hp.get("task", "classification") == "regression":
        return float(selected_score(output, hp, target_class).detach().cpu())
    return float(torch.exp(output[0, target_class]).detach().cpu())


def write_modality_perturbations(model, batch, hp, out_dir, target_class):
    with torch.no_grad():
        h_a, h_b, h_e = model._encode_modalities(batch)
        base = predict_from_aligned(model, batch, h_a, h_b, h_e)
        base_score = display_score(base, hp, target_class)

        rows = [{"mask": "none", "score": base_score, "delta_from_base": 0.0}]
        masks = {
            "drug_a_zero": (torch.zeros_like(h_a), h_b, h_e),
            "drug_b_zero": (h_a, torch.zeros_like(h_b), h_e),
            "cell_zero": (h_a, h_b, torch.zeros_like(h_e)),
            "drug_pair_zero": (torch.zeros_like(h_a), torch.zeros_like(h_b), h_e),
            "all_zero": (torch.zeros_like(h_a), torch.zeros_like(h_b), torch.zeros_like(h_e)),
        }
        for name, tensors in masks.items():
            output = predict_from_aligned(model, batch, *tensors)
            score = display_score(output, hp, target_class)
            rows.append({"mask": name, "score": score, "delta_from_base": score - base_score})

    pd.DataFrame(rows).to_csv(out_dir / "modality_perturbation.csv", index=False)


def write_drug_pair_response_curve(model, batch, hp, out_dir, target_class, num_points=21):
    """Intervene on aligned drug representations and trace prediction response."""
    with torch.no_grad():
        h_a, h_b, h_e = model._encode_modalities(batch)
        base = display_score(predict_from_aligned(model, batch, h_a, h_b, h_e), hp, target_class)

        rows = []
        alphas = torch.linspace(0.0, 1.0, num_points, device=h_a.device)
        for alpha in alphas:
            alpha_value = float(alpha.detach().cpu())
            interventions = {
                "drug_pair_retention": (alpha * h_a, alpha * h_b, h_e),
                "drug_a_retention": (alpha * h_a, h_b, h_e),
                "drug_b_retention": (h_a, alpha * h_b, h_e),
            }
            for name, tensors in interventions.items():
                score = display_score(predict_from_aligned(model, batch, *tensors), hp, target_class)
                rows.append(
                    {
                        "intervention": name,
                        "alpha": alpha_value,
                        "score": score,
                        "delta_from_base": score - base,
                    }
                )

    pd.DataFrame(rows).to_csv(out_dir / "drug_pair_response_curve.csv", index=False)


def write_gradient_saliency(model, batch, hp, out_dir, target_class, top_k, gene_names):
    atom_outputs = []

    def atom_hook(_module, _inputs, output):
        output.retain_grad()
        atom_outputs.append(output)

    handle = None
    if getattr(model, "drug_encoder_uses_atom_encoder", False):
        handle = model.drug_encoder.gnn_node.atom_encoder.register_forward_hook(atom_hook)

    batch.expression = batch.expression.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    output = model(batch)
    score = selected_score(output, hp, target_class)
    score.backward()

    expr_grad = batch.expression.grad.detach()[0].abs().cpu()
    expr_rows = []
    for idx, value in enumerate(expr_grad.tolist()):
        name = gene_names[idx] if gene_names is not None and idx < len(gene_names) else f"feature_{idx}"
        expr_rows.append({"feature_index": idx, "feature_name": name, "gradient_abs": value})
    pd.DataFrame(expr_rows).sort_values("gradient_abs", ascending=False).head(top_k).to_csv(
        out_dir / "expression_gradient_importance.csv", index=False
    )

    atom_rows = []
    for drug_slot, tensor in zip(["drug_a", "drug_b"], atom_outputs[:2]):
        grad = tensor.grad
        if grad is None:
            continue
        saliency = (grad.detach() * tensor.detach()).abs().sum(dim=1).cpu()
        batch_vec = getattr(batch, f"x_{drug_slot[-1]}_batch", None)
        if batch_vec is None:
            batch_vec = torch.zeros(saliency.numel(), dtype=torch.long)
        local_ids = torch.zeros_like(batch_vec.cpu())
        for graph_id in batch_vec.cpu().unique().tolist():
            mask = batch_vec.cpu() == graph_id
            local_ids[mask] = torch.arange(int(mask.sum()))
        for node_idx, value in enumerate(saliency.tolist()):
            atom_rows.append(
                {
                    "drug_slot": drug_slot,
                    "global_node_index": node_idx,
                    "local_atom_index": int(local_ids[node_idx]),
                    "saliency_grad_x_activation": value,
                }
            )
    if handle is not None:
        handle.remove()

    if atom_rows:
        pd.DataFrame(atom_rows).sort_values("saliency_grad_x_activation", ascending=False).head(top_k).to_csv(
            out_dir / "atom_embedding_saliency.csv", index=False
        )


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else exp_dir / "modelstates" / "best_model.pt"
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir / "explanations" / f"sample_{args.sample_index}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, hp = load_model(exp_dir, checkpoint, device)

    dataset = MoleculeDataset(root=args.dataset_root, dataset="tdcSynergy")
    loader = DataLoader([dataset[args.sample_index]], batch_size=1, shuffle=False, follow_batch=["x_a", "x_b"])
    batch = next(iter(loader)).to(device)
    batch.x_a = batch.x_a.long()
    batch.x_b = batch.x_b.long()

    gene_names = load_gene_names(args.gene_names)
    write_modality_perturbations(model, batch, hp, out_dir, args.target_class)
    write_drug_pair_response_curve(model, batch, hp, out_dir, args.target_class)
    write_gradient_saliency(model, batch, hp, out_dir, args.target_class, args.top_k, gene_names)

    print(f"Wrote explanation outputs to {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
