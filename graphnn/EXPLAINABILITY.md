# Explainability tools

This project now has two practical explainability entry points.

## 1. KAN function curves and channel importance

Use this on any checkpoint that contains KAN parameters. It works with the
existing KAGNN checkpoints that store `gnn_model_state_dict`.

```bash
python3 graphnn/explain_kan_functions.py \
  --checkpoint graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1/experiments/<run>/<exp>/modelstates/best_model.pt \
  --output-dir graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1/experiments/<run>/<exp>/explanations/kan \
  --top-k 24
```

Outputs:

- `kan_channel_importance.csv`: every KAN input-output channel pair.
- `kan_input_channel_summary.csv`: importance aggregated by input channel.
- `kan_layer_summary.csv`: nonlinear strength and roughness by KAN layer.
- `kan_curves_top_channels.csv`: sampled curve values for the strongest channels.
- `curves/*.png`: plotted KAN functions.

Interpretation:

- `harmonic_l2` is the non-constant Fourier strength or spline coefficient variation.
- `roughness` is a proxy for function complexity.
- Large `input_channel` scores identify hidden dimensions that the KAN relies on most.

## 2. Single-sample drug-drug-cell explanations

This is for `DeepDDS_Hypergraph` experiments. New hypergraph training runs now
save `modelstates/best_model.pt` with `model_state_dict`, which this script
loads directly.

```bash
python3 graphnn/explain_synergy_sample.py \
  --exp-dir graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1/experiments/<run>/<exp> \
  --dataset-root graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1 \
  --sample-index 0 \
  --device cpu
```

Outputs:

- `modality_perturbation.csv`: prediction changes after masking drug A, drug B,
  cell line, both drugs, or all three aligned embeddings.
- `expression_gradient_importance.csv`: expression feature gradient importance.
- `atom_embedding_saliency.csv`: atom-level saliency over the initial atom
  embeddings for drug A and drug B.

Optional gene labels:

```bash
python3 graphnn/explain_synergy_sample.py ... --gene-names path/to/gene_names.txt
```

