import argparse
import copy
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, HypergraphConv, global_max_pool, global_mean_pool


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(MODEL_DIR)
sys.path.append(MODEL_DIR)

from drug_util import GraphDataset, collate  # noqa: E402
from process_data import getData  # noqa: E402
from similarity import get_Cosin_Similarity, get_pvalue_matrix  # noqa: E402
from utils import reset, set_seed_all  # noqa: E402


warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def scatter_mean(src, index, dim_size):
    out = src.new_zeros((dim_size, src.size(-1)))
    count = src.new_zeros((dim_size, 1))
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones((index.numel(), 1), device=src.device, dtype=src.dtype))
    return out / count.clamp_min(1.0)


class FourierKANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_frequencies=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, 2 * num_frequencies + 1) * 0.05
        )
        self.base = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("frequencies", torch.arange(1, num_frequencies + 1).float())

    def forward(self, x):
        x_bounded = torch.tanh(x)
        x_expanded = x_bounded.unsqueeze(1).unsqueeze(-1)
        angles = x_expanded * self.frequencies * np.pi
        basis = torch.cat(
            [torch.ones_like(x_expanded[..., :1]), torch.cos(angles), torch.sin(angles)],
            dim=-1,
        )
        kan_out = torch.einsum("boif,oif->bo", basis, self.coeffs)
        return self.norm(kan_out + self.base(x))


class KANHypergraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_frequencies=3):
        super().__init__()
        self.out_channels = out_channels
        self.edge_kan = FourierKANLinear(in_channels, out_channels, num_frequencies)

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            return x.new_zeros((x.size(0), self.out_channels))

        node_idx = edge_index[0].long()
        edge_idx = edge_index[1].long()
        num_edges = int(edge_idx.max().item()) + 1
        edge_embed = scatter_mean(x[node_idx], edge_idx, num_edges)
        edge_embed = self.edge_kan(edge_embed)
        return scatter_mean(edge_embed[edge_idx], node_idx, x.size(0))


class HgnnEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.batch1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, out_channels)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        if edge_index.numel() == 0:
            return x.new_zeros((x.size(0), self.out_channels))
        x = self.act(self.conv1(x, edge_index))
        x = self.batch1(x)
        x = self.drop_out(x)
        return self.act(self.conv2(x, edge_index))


class KANHgnnEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=256, num_frequencies=3):
        super().__init__()
        self.conv1 = KANHypergraphConv(in_channels, hidden_channels, num_frequencies)
        self.batch1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = KANHypergraphConv(hidden_channels, out_channels, num_frequencies)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        x = self.act(self.conv1(x, edge_index))
        x = self.batch1(x)
        x = self.drop_out(x)
        return self.act(self.conv2(x, edge_index))


class BioEncoder(nn.Module):
    def __init__(self, dim_drug, dim_cellline, output, use_GMP=True):
        super().__init__()
        self.use_GMP = use_GMP
        self.conv1 = GCNConv(dim_drug, 128)
        self.batch_conv1 = nn.BatchNorm1d(128)
        self.conv2 = GCNConv(128, output)
        self.batch_conv2 = nn.BatchNorm1d(output)
        self.fc_cell1 = nn.Linear(dim_cellline, 128)
        self.batch_cell1 = nn.BatchNorm1d(128)
        self.fc_cell2 = nn.Linear(128, output)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.ReLU()

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data):
        x_drug = self.conv1(drug_feature, drug_adj)
        x_drug = self.batch_conv1(self.act(x_drug))
        x_drug = self.drop_out(x_drug)
        x_drug = self.conv2(x_drug, drug_adj)
        x_drug = self.batch_conv2(self.act(x_drug))
        if self.use_GMP:
            x_drug = global_max_pool(x_drug, ibatch)
        else:
            x_drug = global_mean_pool(x_drug, ibatch)

        x_cellline = torch.tanh(self.fc_cell1(gexpr_data))
        x_cellline = self.batch_cell1(x_cellline)
        x_cellline = self.drop_out(x_cellline)
        x_cellline = self.act(self.fc_cell2(x_cellline))
        return x_drug, x_cellline


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.batch1 = nn.BatchNorm1d(in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, in_channels // 4)
        self.batch2 = nn.BatchNorm1d(in_channels // 4)
        self.fc3 = nn.Linear(in_channels // 4, 1)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, graph_embed, druga_id, drugb_id, cellline_id):
        h = torch.cat(
            (graph_embed[druga_id, :], graph_embed[drugb_id, :], graph_embed[cellline_id, :]),
            1,
        )
        h = self.drop_out(self.batch1(self.act(self.fc1(h))))
        h = self.drop_out(self.batch2(self.act(self.fc2(h))))
        return torch.sigmoid(self.fc3(h).squeeze(dim=1))


class HypergraphSynergyAblation(nn.Module):
    def __init__(self, bio_encoder, graph_encoder, decoder, num_drugs, num_clines, bio_dim=100, graph_dim=256):
        super().__init__()
        self.bio_encoder = bio_encoder
        self.graph_encoder = graph_encoder
        self.decoder = decoder
        self.num_drugs = num_drugs
        self.num_clines = num_clines
        self.fallback = nn.Linear(bio_dim, graph_dim)
        self.drug_rec_weight = nn.Parameter(torch.empty(graph_dim, graph_dim))
        self.cline_rec_weight = nn.Parameter(torch.empty(graph_dim, graph_dim))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.bio_encoder)
        reset(self.graph_encoder)
        reset(self.decoder)
        nn.init.xavier_uniform_(self.fallback.weight)
        nn.init.zeros_(self.fallback.bias)
        nn.init.xavier_uniform_(self.drug_rec_weight)
        nn.init.xavier_uniform_(self.cline_rec_weight)

    def forward(self, drug_feature, drug_adj, ibatch, gexpr_data, adj, druga_id, drugb_id, cellline_id):
        drug_embed, cellline_embed = self.bio_encoder(drug_feature, drug_adj, ibatch, gexpr_data)
        merge_embed = torch.cat((drug_embed, cellline_embed), 0)
        graph_embed = self.graph_encoder(merge_embed, adj)

        fallback_embed = self.fallback(merge_embed)
        incident_mask = torch.zeros(merge_embed.size(0), dtype=torch.bool, device=merge_embed.device)
        if adj.numel() > 0:
            incident_mask[torch.unique(adj[0].long())] = True
        graph_embed = torch.where(incident_mask.unsqueeze(1), graph_embed, fallback_embed)

        drug_emb = graph_embed[: self.num_drugs]
        cline_emb = graph_embed[self.num_drugs : self.num_drugs + self.num_clines]
        rec_drug = torch.sigmoid(torch.mm(torch.mm(drug_emb, self.drug_rec_weight), drug_emb.t()))
        rec_cline = torch.sigmoid(torch.mm(torch.mm(cline_emb, self.cline_rec_weight), cline_emb.t()))
        pred = self.decoder(graph_embed, druga_id, drugb_id, cellline_id)
        return pred, rec_drug, rec_cline, incident_mask


def load_data(dataset, threshold):
    cline_fea, drug_fea, drug_smiles_fea, gene_data, synergy = getData(dataset)
    synergy = np.array(synergy, dtype=np.float32)
    labels = (synergy[:, 3] >= threshold).astype(np.float32)
    synergy[:, 3] = labels

    cline_fea = torch.from_numpy(cline_fea).to(device)
    drug_sim_matrix = np.array(get_Cosin_Similarity(drug_smiles_fea))
    cline_sim_matrix = np.array(get_pvalue_matrix(np.array(gene_data, dtype="float32")))
    drug_sim_mat = torch.from_numpy(drug_sim_matrix).float().to(device)
    cline_sim_mat = torch.from_numpy(cline_sim_matrix).float().to(device)
    return drug_fea, cline_fea, synergy, drug_sim_mat, cline_sim_mat


def stratify_or_none(labels):
    _, counts = np.unique(labels, return_counts=True)
    return labels if len(counts) == 2 and counts.min() >= 2 else None


def random_split(synergy, seed, test_frac=0.1, val_frac=0.1):
    indices = np.arange(len(synergy))
    labels = synergy[:, 3].astype(int)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_frac,
        random_state=seed,
        shuffle=True,
        stratify=stratify_or_none(labels),
    )
    relative_val_frac = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val_frac,
        random_state=seed,
        shuffle=True,
        stratify=stratify_or_none(labels[train_val_idx]),
    )
    return {"train": np.sort(train_idx), "valid": np.sort(val_idx), "test": np.sort(test_idx)}


def build_drug_index(drug_a, drug_b, indices):
    drug_to_indices = {}
    for idx in indices:
        idx = int(idx)
        drug_to_indices.setdefault(int(drug_a[idx]), set()).add(idx)
        drug_to_indices.setdefault(int(drug_b[idx]), set()).add(idx)
    return drug_to_indices


def score_exposure(count, pos_count, target_count, target_pos_rate):
    if target_count <= 0:
        return float(count)
    count_error = abs(count - target_count) / float(target_count)
    if count > target_count:
        count_error *= 1.05
    pos_rate = pos_count / float(count) if count else 0.0
    return count_error + 0.25 * abs(pos_rate - target_pos_rate)


def select_drugs_for_target(drug_to_indices, candidate_drugs, labels, target_count, rng, target_pos_rate):
    candidate_drugs = [int(d) for d in candidate_drugs if int(d) in drug_to_indices]
    rng.shuffle(candidate_drugs)
    selected_drugs = set()
    selected_indices = set()
    selected_pos = 0.0
    current_score = score_exposure(0, 0.0, target_count, target_pos_rate)

    while candidate_drugs:
        best = None
        for drug in candidate_drugs:
            marginal = drug_to_indices[drug] - selected_indices
            if not marginal:
                continue
            marginal_list = list(marginal)
            new_count = len(selected_indices) + len(marginal_list)
            new_pos = selected_pos + float(labels[marginal_list].sum())
            score = score_exposure(new_count, new_pos, target_count, target_pos_rate)
            tie_break = (score, abs(new_count - target_count), len(marginal_list), drug)
            if best is None or tie_break < best[0]:
                best = (tie_break, drug, marginal, new_pos)
        if best is None:
            break

        best_score = best[0][0]
        if selected_indices and best_score >= current_score:
            break

        _, drug, marginal, new_pos = best
        selected_drugs.add(drug)
        selected_indices.update(marginal)
        selected_pos = new_pos
        current_score = best_score
        if len(selected_indices) >= target_count and current_score < 0.01:
            break

    return selected_drugs, selected_indices


def cold_drug_split(synergy, seed, test_frac=0.1, val_frac=0.1):
    reference = random_split(synergy, seed, test_frac, val_frac)
    target_valid = len(reference["valid"])
    target_test = len(reference["test"])

    drug_a = synergy[:, 0].astype(int)
    drug_b = synergy[:, 1].astype(int)
    labels = synergy[:, 3].astype(int)
    all_indices = set(range(len(synergy)))
    all_drugs = set(np.unique(np.concatenate([drug_a, drug_b])).astype(int).tolist())
    rng = np.random.RandomState(seed)
    target_pos_rate = float(labels.mean()) if len(labels) else 0.0

    drug_to_indices = build_drug_index(drug_a, drug_b, all_indices)
    test_drugs, test_indices = select_drugs_for_target(
        drug_to_indices,
        list(all_drugs),
        labels,
        target_test,
        rng,
        target_pos_rate,
    )

    remaining = all_indices - test_indices
    valid_drug_to_indices = build_drug_index(drug_a, drug_b, remaining)
    valid_drugs, valid_indices = select_drugs_for_target(
        valid_drug_to_indices,
        list(all_drugs - test_drugs),
        labels,
        target_valid,
        rng,
        target_pos_rate,
    )

    train_indices = all_indices - test_indices - valid_indices
    return {
        "train": np.array(sorted(train_indices), dtype=np.int64),
        "valid": np.array(sorted(valid_indices), dtype=np.int64),
        "test": np.array(sorted(test_indices), dtype=np.int64),
        "heldout_test_drugs": sorted(test_drugs),
        "heldout_valid_drugs": sorted(valid_drugs),
    }


def build_synergy_graph(synergy_train):
    pos_edges = synergy_train[synergy_train[:, 3] == 1, 0:3].astype(np.int64)
    if len(pos_edges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    synergy_edge = pos_edges.reshape(1, -1)
    edge_ids = np.repeat(np.arange(len(pos_edges), dtype=np.int64), 3).reshape(1, -1)
    synergy_graph = np.concatenate((synergy_edge, edge_ids), axis=0)
    return torch.from_numpy(synergy_graph).long().to(device)


def make_tensors(split_data):
    label = torch.from_numpy(split_data[:, 3].astype(np.float32)).to(device)
    index = torch.from_numpy(split_data[:, 0:3].astype(np.int64)).long().to(device)
    return index, label


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred)
    y_hat = (y_pred >= 0.5).astype(int)
    thresholds = np.unique(y_pred)
    best_f1 = 0.0
    best_acc = accuracy_score(y_true, y_hat)
    best_threshold = 0.5
    if thresholds.size:
        if thresholds.size > 1000:
            thresholds = np.quantile(thresholds, np.linspace(0, 1, 1000))
        for threshold in thresholds:
            candidate = (y_pred >= threshold).astype(int)
            candidate_f1 = f1_score(y_true, candidate, zero_division=0)
            if candidate_f1 > best_f1:
                best_f1 = candidate_f1
                best_acc = accuracy_score(y_true, candidate)
                best_threshold = float(threshold)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return {
        "auc": roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else np.nan,
        "aupr": average_precision_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else np.nan,
        "aupr_trapz": -np.trapz(precision, recall) if len(np.unique(y_true)) == 2 else np.nan,
        "f1": f1_score(y_true, y_hat, zero_division=0),
        "acc": accuracy_score(y_true, y_hat),
        "best_f1": best_f1,
        "best_acc": best_acc,
        "best_threshold": best_threshold,
    }


def forward_once(model, drug_set, cline_set, synergy_graph, index):
    for drug, cline in zip(drug_set, cline_set):
        return model(
            drug.x.to(device),
            drug.edge_index.to(device),
            drug.batch.to(device),
            cline[0].to(device),
            synergy_graph,
            index[:, 0],
            index[:, 1],
            index[:, 2],
        )
    raise RuntimeError("empty drug/cell line loader")


def task_loss_value(pred, label, pos_weight):
    if pos_weight is None:
        return nn.functional.binary_cross_entropy(pred, label)
    sample_weight = torch.where(label > 0.5, pos_weight, torch.ones_like(label))
    return nn.functional.binary_cross_entropy(pred, label, weight=sample_weight)


def evaluate(model, drug_set, cline_set, synergy_graph, index, label, alpha, pos_weight, rec_loss, drug_sim, cline_sim):
    model.eval()
    with torch.no_grad():
        pred, rec_drug, rec_cline, incident_mask = forward_once(model, drug_set, cline_set, synergy_graph, index)
        task_loss = task_loss_value(pred, label, pos_weight)
        aux_loss = rec_loss(rec_drug, drug_sim) + rec_loss(rec_cline, cline_sim)
        loss = (1 - alpha) * task_loss + alpha * aux_loss
    metrics = binary_metrics(label.detach().cpu().numpy(), pred.detach().cpu().numpy())
    return metrics, float(loss.item()), pred.detach().cpu().numpy(), incident_mask.detach().cpu().numpy()


def train_epoch(model, optimizer, drug_set, cline_set, synergy_graph, index, label, alpha, pos_weight, rec_loss, drug_sim, cline_sim):
    model.train()
    optimizer.zero_grad()
    pred, rec_drug, rec_cline, _ = forward_once(model, drug_set, cline_set, synergy_graph, index)
    task_loss = task_loss_value(pred, label, pos_weight)
    aux_loss = rec_loss(rec_drug, drug_sim) + rec_loss(rec_cline, cline_sim)
    loss = (1 - alpha) * task_loss + alpha * aux_loss
    loss.backward()
    optimizer.step()
    metrics = binary_metrics(label.detach().cpu().numpy(), pred.detach().cpu().numpy())
    return metrics, float(loss.item())


def build_model(encoder_name, cline_dim, num_drugs, num_clines, kan_frequencies):
    bio_dim = 100
    graph_dim = 256
    if encoder_name == "hgnn":
        graph_encoder = HgnnEncoder(bio_dim, graph_dim)
    elif encoder_name == "kan":
        graph_encoder = KANHgnnEncoder(bio_dim, graph_dim, num_frequencies=kan_frequencies)
    else:
        raise ValueError(f"unknown encoder: {encoder_name}")

    return HypergraphSynergyAblation(
        BioEncoder(dim_drug=75, dim_cellline=cline_dim, output=bio_dim),
        graph_encoder,
        Decoder(in_channels=graph_dim * 3),
        num_drugs=num_drugs,
        num_clines=num_clines,
        bio_dim=bio_dim,
        graph_dim=graph_dim,
    ).to(device)


def summarize_split(split_name, synergy, partition, synergy_graph):
    train = synergy[partition["train"]]
    valid = synergy[partition["valid"]]
    test = synergy[partition["test"]]
    train_drugs = set(train[:, 0].astype(int).tolist()) | set(train[:, 1].astype(int).tolist())
    test_drugs = set(test[:, 0].astype(int).tolist()) | set(test[:, 1].astype(int).tolist())
    heldout_test_drugs = set(partition.get("heldout_test_drugs", []))
    heldout_valid_drugs = set(partition.get("heldout_valid_drugs", []))
    cold_overlap = len(train_drugs & test_drugs)
    return {
        "split": split_name,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "train_pos_rate": float(train[:, 3].mean()) if len(train) else np.nan,
        "valid_pos_rate": float(valid[:, 3].mean()) if len(valid) else np.nan,
        "test_pos_rate": float(test[:, 3].mean()) if len(test) else np.nan,
        "n_hyperedges": int(synergy_graph.size(1) // 3),
        "train_test_drug_overlap": cold_overlap,
        "n_heldout_test_drugs": len(heldout_test_drugs),
        "n_heldout_valid_drugs": len(heldout_valid_drugs),
        "heldout_test_train_overlap": len(heldout_test_drugs & train_drugs),
        "heldout_valid_train_overlap": len(heldout_valid_drugs & train_drugs),
    }


def run_one(args, encoder_name, split_name, partition, data_bundle):
    drug_feature, cline_feature, synergy, drug_sim, cline_sim = data_bundle
    num_drugs = len(drug_feature)
    num_clines = int(cline_feature.shape[0])

    train_data = synergy[partition["train"]]
    valid_data = synergy[partition["valid"]]
    test_data = synergy[partition["test"]]
    synergy_graph = build_synergy_graph(train_data)

    drug_set = Data.DataLoader(
        dataset=GraphDataset(graphs_dict=drug_feature),
        collate_fn=collate,
        batch_size=len(drug_feature),
        shuffle=False,
    )
    cline_set = Data.DataLoader(
        dataset=Data.TensorDataset(cline_feature),
        batch_size=len(cline_feature),
        shuffle=False,
    )

    index_train, label_train = make_tensors(train_data)
    index_valid, label_valid = make_tensors(valid_data)
    index_test, label_test = make_tensors(test_data)

    model = build_model(encoder_name, cline_feature.shape[-1], num_drugs, num_clines, args.kan_frequencies)
    rec_loss = nn.BCELoss()
    if args.class_weight:
        pos = label_train.sum().clamp_min(1.0)
        neg = (label_train.numel() - label_train.sum()).clamp_min(1.0)
        pos_weight = (neg / pos).to(device)
    else:
        pos_weight = None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -np.inf
    best_epoch = -1
    best_state = None
    patience = 0

    for epoch in range(args.epochs):
        train_metric, train_loss = train_epoch(
            model,
            optimizer,
            drug_set,
            cline_set,
            synergy_graph,
            index_train,
            label_train,
            args.alpha,
            pos_weight,
            rec_loss,
            drug_sim,
            cline_sim,
        )
        val_metric, val_loss, _, _ = evaluate(
            model,
            drug_set,
            cline_set,
            synergy_graph,
            index_valid,
            label_valid,
            args.alpha,
            pos_weight,
            rec_loss,
            drug_sim,
            cline_sim,
        )
        score = val_metric[args.selection_metric]
        if np.isnan(score):
            score = val_metric["aupr"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(
                f"[{split_name}/{encoder_name}] epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"train_auc={train_metric['auc']:.4f} val_auc={val_metric['auc']:.4f} "
                f"val_aupr={val_metric['aupr']:.4f} val_best_f1={val_metric['best_f1']:.4f}"
            )

        if args.early_stopping > 0 and patience >= args.early_stopping:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_metric, val_loss, val_pred, incident_mask = evaluate(
        model,
        drug_set,
        cline_set,
        synergy_graph,
        index_valid,
        label_valid,
        args.alpha,
        pos_weight,
        rec_loss,
        drug_sim,
        cline_sim,
    )
    test_metric, test_loss, test_pred, _ = evaluate(
        model,
        drug_set,
        cline_set,
        synergy_graph,
        index_test,
        label_test,
        args.alpha,
        pos_weight,
        rec_loss,
        drug_sim,
        cline_sim,
    )

    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    pred_prefix = f"{args.dataset}_{split_name}_{encoder_name}_seed{args.seed}"
    np.savetxt(os.path.join(output_dir, pred_prefix + "_val_pred.txt"), val_pred)
    np.savetxt(os.path.join(output_dir, pred_prefix + "_test_pred.txt"), test_pred)

    split_summary = summarize_split(split_name, synergy, partition, synergy_graph)
    result = {
        "dataset": args.dataset,
        "split": split_name,
        "encoder": encoder_name,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "val_auc": val_metric["auc"],
        "val_aupr": val_metric["aupr"],
        "val_f1": val_metric["f1"],
        "val_acc": val_metric["acc"],
        "val_best_f1": val_metric["best_f1"],
        "val_best_acc": val_metric["best_acc"],
        "val_best_threshold": val_metric["best_threshold"],
        "test_auc": test_metric["auc"],
        "test_aupr": test_metric["aupr"],
        "test_aupr_trapz": test_metric["aupr_trapz"],
        "test_f1": test_metric["f1"],
        "test_acc": test_metric["acc"],
        "test_best_f1": test_metric["best_f1"],
        "test_best_acc": test_metric["best_acc"],
        "test_best_threshold": test_metric["best_threshold"],
        "n_incident_nodes": int(incident_mask.sum()),
        "n_total_nodes": int(incident_mask.shape[0]),
        "n_nonincident_drugs": int((~incident_mask[:num_drugs]).sum()),
        "n_nonincident_clines": int((~incident_mask[num_drugs : num_drugs + num_clines]).sum()),
    }
    result.update(split_summary)
    print(
        f"[done] {split_name}/{encoder_name}: best_epoch={best_epoch}, "
        f"test_auc={test_metric['auc']:.4f}, test_aupr={test_metric['aupr']:.4f}, "
        f"test_f1={test_metric['f1']:.4f}, test_acc={test_metric['acc']:.4f}"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ALMANAC", choices=["ALMANAC", "ONEIL"])
    parser.add_argument("--splits", default="random,cold_drug")
    parser.add_argument("--encoders", default="hgnn,kan")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--early_stopping", type=int, default=15)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--threshold", type=float, default=30.0)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--kan_frequencies", type=int, default=3)
    parser.add_argument("--selection_metric", default="aupr", choices=["aupr", "auc", "best_f1"])
    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--output_dir", default="results_kan_ablation")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed_all(args.seed)
    print(f"device={device}, dataset={args.dataset}, seed={args.seed}")
    data_bundle = load_data(args.dataset, args.threshold)
    synergy = data_bundle[2]

    partitions = {}
    for split_name in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split_name == "random":
            partitions[split_name] = random_split(synergy, args.seed, args.test_frac, args.val_frac)
        elif split_name == "cold_drug":
            partitions[split_name] = cold_drug_split(synergy, args.seed, args.test_frac, args.val_frac)
        else:
            raise ValueError(f"unknown split: {split_name}")

    results = []
    for split_name, partition in partitions.items():
        for encoder_name in [e.strip() for e in args.encoders.split(",") if e.strip()]:
            set_seed_all(args.seed)
            results.append(run_one(args, encoder_name, split_name, partition, data_bundle))

    output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.csv")
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
