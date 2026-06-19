import argparse
import copy
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.model_selection import KFold


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(MODEL_DIR)
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from drug_util import GraphDataset, collate  # noqa: E402
from model import BioEncoder, Decoder, HgnnEncoder, HypergraphSynergy  # noqa: E402
from process_data import getData  # noqa: E402
from similarity import get_Cosin_Similarity, get_pvalue_matrix  # noqa: E402
from utils import metrics_graph, set_seed_all  # noqa: E402


class FourierKANLinear(nn.Module):
    def __init__(self, in_features, out_features, num_frequencies=3):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.coeffs = nn.Parameter(
            torch.randn(out_features, in_features, 2 * num_frequencies + 1) * 0.05
        )
        self.norm = nn.LayerNorm(out_features)
        self.register_buffer("freq", torch.arange(1, num_frequencies + 1).float())

    def forward(self, x):
        x_limited = torch.tanh(x)
        x_expanded = x_limited.unsqueeze(1).unsqueeze(-1)
        angles = x_expanded * self.freq * np.pi
        basis = torch.cat(
            [torch.ones_like(x_expanded[..., :1]), torch.cos(angles), torch.sin(angles)],
            dim=-1,
        )
        kan_out = torch.einsum("boif,oif->bo", basis, self.coeffs)
        return self.norm(self.base(x) + kan_out)


def scatter_mean(src, index, dim_size):
    out = src.new_zeros((dim_size, src.size(-1)))
    count = src.new_zeros((dim_size, 1))
    out.index_add_(0, index, src)
    count.index_add_(0, index, torch.ones((index.numel(), 1), device=src.device, dtype=src.dtype))
    return out / count.clamp_min(1.0)


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


class KANHgnnEncoder(nn.Module):
    """Drop-in replacement for the original HgnnEncoder using KAN edge functions."""

    def __init__(self, in_channels, out_channels, num_frequencies=3):
        super().__init__()
        self.conv1 = KANHypergraphConv(in_channels, 256, num_frequencies)
        self.batch1 = nn.BatchNorm1d(256)
        self.conv2 = KANHypergraphConv(256, 256, num_frequencies)
        self.batch2 = nn.BatchNorm1d(256)
        self.conv3 = KANHypergraphConv(256, out_channels, num_frequencies)
        self.act = nn.ReLU()

    def forward(self, x, edge):
        x = self.batch1(self.act(self.conv1(x, edge)))
        x = self.batch2(self.act(self.conv2(x, edge)))
        x = self.act(self.conv3(x, edge))
        return x


def load_data(dataset, device):
    cline_fea, drug_fea, drug_smiles_fea, gene_data, synergy = getData(dataset)
    cline_fea = torch.from_numpy(cline_fea).to(device)
    for row in synergy:
        row[3] = 1 if row[3] >= 30 else 0
    drug_sim_matrix = np.array(get_Cosin_Similarity(drug_smiles_fea))
    cline_sim_matrix = np.array(get_pvalue_matrix(np.array(gene_data, dtype="float32")))
    drug_sim = torch.from_numpy(drug_sim_matrix).float().to(device)
    cline_sim = torch.from_numpy(cline_sim_matrix).float().to(device)
    return drug_fea, cline_fea, np.array(synergy), drug_sim, cline_sim


def random_outer_split(synergy, seed, output_prefix):
    synergy_pos = pd.DataFrame([i for i in synergy if i[3] == 1])
    synergy_neg = pd.DataFrame([i for i in synergy if i[3] == 0])
    train_size = 0.9
    synergy_cv_pos, synergy_test_pos = np.split(
        np.array(synergy_pos.sample(frac=1, random_state=seed)),
        [int(train_size * len(synergy_pos))],
    )
    synergy_cv_neg, synergy_test_neg = np.split(
        np.array(synergy_neg.sample(frac=1, random_state=seed)),
        [int(train_size * len(synergy_neg))],
    )
    synergy_cv = np.concatenate((synergy_cv_neg, synergy_cv_pos), axis=0)
    synergy_test = np.concatenate((synergy_test_neg, synergy_test_pos), axis=0)
    np.random.shuffle(synergy_cv)
    np.random.shuffle(synergy_test)
    np.savetxt(output_prefix + "test_y_true.txt", synergy_test[:, 3])
    return synergy_cv, synergy_test


def _drug_to_indices(drug_a, drug_b, indices):
    out = {}
    for idx in indices:
        idx = int(idx)
        out.setdefault(int(drug_a[idx]), set()).add(idx)
        out.setdefault(int(drug_b[idx]), set()).add(idx)
    return out


def _score(count, pos_count, target_count, target_pos_rate):
    if target_count <= 0:
        return float(count)
    count_error = abs(count - target_count) / float(target_count)
    pos_rate = pos_count / float(count) if count else 0.0
    return count_error + 0.25 * abs(pos_rate - target_pos_rate)


def _select_drugs(drug_to_indices, candidate_drugs, labels, target_count, rng, target_pos_rate):
    candidate_drugs = [int(d) for d in candidate_drugs if int(d) in drug_to_indices]
    rng.shuffle(candidate_drugs)
    selected_drugs = set()
    selected_indices = set()
    selected_pos = 0.0
    current_score = _score(0, 0.0, target_count, target_pos_rate)

    while candidate_drugs:
        best = None
        for drug in candidate_drugs:
            marginal = drug_to_indices[drug] - selected_indices
            if not marginal:
                continue
            marginal_list = list(marginal)
            new_count = len(selected_indices) + len(marginal_list)
            new_pos = selected_pos + float(labels[marginal_list].sum())
            score = _score(new_count, new_pos, target_count, target_pos_rate)
            key = (score, abs(new_count - target_count), len(marginal_list), drug)
            if best is None or key < best[0]:
                best = (key, drug, marginal, new_pos)
        if best is None or (selected_indices and best[0][0] >= current_score):
            break
        _, drug, marginal, selected_pos = best
        selected_drugs.add(drug)
        selected_indices.update(marginal)
        current_score = best[0][0]
    return selected_drugs, selected_indices


def cold_drug_split(synergy, seed, output_prefix, test_frac=0.1, val_frac=0.1):
    labels = synergy[:, 3].astype(int)
    all_indices = set(range(len(synergy)))
    target_test = int(round(len(synergy) * test_frac))
    target_val = int(round(len(synergy) * val_frac))
    drug_a = synergy[:, 0].astype(int)
    drug_b = synergy[:, 1].astype(int)
    all_drugs = set(np.unique(np.concatenate([drug_a, drug_b])).astype(int).tolist())
    rng = np.random.RandomState(seed)
    target_pos_rate = float(labels.mean())

    test_drugs, test_indices = _select_drugs(
        _drug_to_indices(drug_a, drug_b, all_indices),
        list(all_drugs),
        labels,
        target_test,
        rng,
        target_pos_rate,
    )
    remaining = all_indices - test_indices
    valid_drugs, valid_indices = _select_drugs(
        _drug_to_indices(drug_a, drug_b, remaining),
        list(all_drugs - test_drugs),
        labels,
        target_val,
        rng,
        target_pos_rate,
    )
    train_indices = sorted(all_indices - test_indices - valid_indices)
    synergy_train = synergy[train_indices]
    synergy_validation = synergy[sorted(valid_indices)]
    synergy_test = synergy[sorted(test_indices)]
    np.savetxt(output_prefix + "test_y_true.txt", synergy_test[:, 3])
    train_drugs = set(synergy_train[:, 0].astype(int)) | set(synergy_train[:, 1].astype(int))
    print(
        "cold split:",
        f"heldout_test_drugs={len(test_drugs)}",
        f"heldout_valid_drugs={len(valid_drugs)}",
        f"heldout_test_train_overlap={len(test_drugs & train_drugs)}",
    )
    return synergy_train, synergy_validation, synergy_test


def build_hypergraph(synergy_train, device):
    edge_data = synergy_train[synergy_train[:, 3] == 1, 0:3].astype(np.int64)
    synergy_edge = edge_data.reshape(1, -1)
    index_num = np.expand_dims(np.arange(len(edge_data)), axis=-1)
    synergy_num = np.concatenate((index_num, index_num, index_num), axis=1).reshape(1, -1)
    return torch.from_numpy(np.concatenate((synergy_edge, synergy_num), axis=0)).long().to(device)


def make_tensors(synergy_part, device):
    label = torch.from_numpy(np.array(synergy_part[:, 3], dtype="float32")).to(device)
    index = torch.from_numpy(synergy_part[:, 0:3]).long().to(device)
    return index, label


def build_model(encoder_name, cline_dim, kan_freq, device):
    if encoder_name == "hypergraph":
        graph_encoder = HgnnEncoder(in_channels=100, out_channels=256)
    elif encoder_name == "kan":
        graph_encoder = KANHgnnEncoder(in_channels=100, out_channels=256, num_frequencies=kan_freq)
    else:
        raise ValueError(f"unknown encoder: {encoder_name}")
    return HypergraphSynergy(
        BioEncoder(dim_drug=75, dim_cellline=cline_dim, output=100),
        graph_encoder,
        Decoder(in_channels=768),
    ).to(device)


def forward_once(model, drug_set, cline_set, synergy_graph, index, device):
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
    raise RuntimeError("empty loader")


def train_epoch(model, optimizer, loss_func, drug_set, cline_set, synergy_graph, index, label, alpha, drug_sim, cline_sim, device):
    model.train()
    optimizer.zero_grad()
    pred, rec_drug, rec_cline = forward_once(model, drug_set, cline_set, synergy_graph, index, device)
    loss = loss_func(pred, label)
    loss_rec_1 = loss_func(rec_drug, drug_sim)
    loss_rec_2 = loss_func(rec_cline, cline_sim)
    loss = (1 - alpha) * loss + alpha * (loss_rec_1 + loss_rec_2)
    loss.backward()
    optimizer.step()
    metric = metrics_graph(label.detach().cpu().numpy(), pred.detach().cpu().numpy())
    return metric, loss.item()


def evaluate(model, loss_func, drug_set, cline_set, synergy_graph, index, label, alpha, drug_sim, cline_sim, device):
    model.eval()
    with torch.no_grad():
        pred, rec_drug, rec_cline = forward_once(model, drug_set, cline_set, synergy_graph, index, device)
        loss = loss_func(pred, label)
        loss_rec_1 = loss_func(rec_drug, drug_sim)
        loss_rec_2 = loss_func(rec_cline, cline_sim)
        loss = (1 - alpha) * loss + alpha * (loss_rec_1 + loss_rec_2)
    metric = metrics_graph(label.detach().cpu().numpy(), pred.detach().cpu().numpy())
    return metric, loss.item(), pred.detach().cpu().numpy()


def run_fold(args, fold_num, split_name, encoder_name, data_bundle, output_prefix, device):
    drug_feature, cline_feature, synergy, drug_sim, cline_sim = data_bundle
    if split_name == "random":
        synergy_cv, synergy_test = random_outer_split(synergy, args.seed, output_prefix)
        cv_data = synergy_cv
        splits = list(KFold(n_splits=5, shuffle=True, random_state=args.seed).split(cv_data))
        train_index, validation_index = splits[fold_num]
        synergy_train, synergy_validation = cv_data[train_index], cv_data[validation_index]
    elif split_name == "cold_drug":
        synergy_train, synergy_validation, synergy_test = cold_drug_split(
            synergy, args.seed + fold_num, output_prefix
        )
    else:
        raise ValueError(split_name)

    np.savetxt(output_prefix + f"val_{fold_num}_true.txt", synergy_validation[:, 3])
    index_train, label_train = make_tensors(synergy_train, device)
    index_validation, label_validation = make_tensors(synergy_validation, device)
    index_test, label_test = make_tensors(synergy_test, device)
    synergy_graph = build_hypergraph(synergy_train, device)

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

    model = build_model(encoder_name, cline_feature.shape[-1], args.kan_freq, device)
    loss_func = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    best_metric = [0, 0, 0, 0]
    best_epoch = 0
    best_state = None

    for epoch in range(args.epochs):
        train_metric, train_loss = train_epoch(
            model, optimizer, loss_func, drug_set, cline_set, synergy_graph,
            index_train, label_train, args.alpha, drug_sim, cline_sim, device
        )
        val_metric, val_loss, _ = evaluate(
            model, loss_func, drug_set, cline_set, synergy_graph,
            index_validation, label_validation, args.alpha, drug_sim, cline_sim, device
        )
        if val_metric[0] > best_metric[0]:
            best_metric = val_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            print(
                f"[{split_name}/{encoder_name}/fold{fold_num}] epoch={epoch:05d} "
                f"train_loss={train_loss:.6f} train_auc={train_metric[0]:.6f} "
                f"train_aupr={train_metric[1]:.6f} val_loss={val_loss:.6f} "
                f"val_auc={val_metric[0]:.6f} val_aupr={val_metric[1]:.6f} "
                f"val_f1={val_metric[2]:.6f} val_acc={val_metric[3]:.6f}",
                flush=True,
            )

    model.load_state_dict(best_state)
    val_metric, _, y_val_pred = evaluate(
        model, loss_func, drug_set, cline_set, synergy_graph,
        index_validation, label_validation, args.alpha, drug_sim, cline_sim, device
    )
    test_metric, _, y_test_pred = evaluate(
        model, loss_func, drug_set, cline_set, synergy_graph,
        index_test, label_test, args.alpha, drug_sim, cline_sim, device
    )
    np.savetxt(output_prefix + f"val_{fold_num}_pred.txt", y_val_pred)
    np.savetxt(output_prefix + f"test_{fold_num}_pred.txt", y_test_pred)
    print(
        f"[done] {split_name}/{encoder_name}/fold{fold_num} best_epoch={best_epoch} "
        f"val={val_metric} test={test_metric}",
        flush=True,
    )
    return best_epoch, val_metric, test_metric


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ALMANAC", choices=["ALMANAC", "ONEIL"])
    parser.add_argument("--split", default="random", choices=["random", "cold_drug"])
    parser.add_argument("--encoder", default="hypergraph", choices=["hypergraph", "kan"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--kan-freq", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--output-dir", default="result_4tests")
    return parser.parse_args()


def main():
    args = parse_args()
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    set_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.split}_{args.encoder}_seed{args.seed}_",
    )
    print(f"device={device} dataset={args.dataset} split={args.split} encoder={args.encoder}", flush=True)
    data_bundle = load_data(args.dataset, device)
    rows = []
    for fold_num in [int(i) for i in args.folds.split(",") if i.strip()]:
        set_seed_all(args.seed + fold_num)
        best_epoch, val_metric, test_metric = run_fold(
            args, fold_num, args.split, args.encoder, data_bundle, prefix, device
        )
        rows.append([fold_num, best_epoch, *val_metric, *test_metric])
    result = pd.DataFrame(
        rows,
        columns=[
            "fold", "best_epoch",
            "val_auc", "val_aupr", "val_f1", "val_acc",
            "test_auc", "test_aupr", "test_f1", "test_acc",
        ],
    )
    result.to_csv(prefix + "summary.csv", index=False)
    mean = result[["test_auc", "test_aupr", "test_f1", "test_acc"]].mean()
    print("mean_test", mean.to_dict(), flush=True)


if __name__ == "__main__":
    main()
