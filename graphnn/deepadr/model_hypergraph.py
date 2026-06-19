"""
DeepDDS with Hypergraph Integration
集成超图神经网络的DeepDDS模型实现
支持多种模式：hypergraph, mean_hypergraph, kan_hypergraph, kan_aggregated
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
import numpy as np

try:
    from .model_gnn_ogb import GATNet, ExpressionNN
    from .model_kagnn import KAGNN
    from .kagnn_conv import KAGCNConv
    from .kan_layers import KANLinear
except ImportError:
    from model_gnn_ogb import GATNet, ExpressionNN
    from model_kagnn import KAGNN
    from kagnn_conv import KAGCNConv
    from kan_layers import KANLinear


class EmbeddingAligner(nn.Module):
    """统一药物和细胞系嵌入维度"""
    def __init__(self, drug_dim=128, cell_dim=256, output_dim=128):
        super().__init__()
        # 药物维度已经是128，保持不变
        if drug_dim == output_dim:
            self.drug_proj = nn.Identity()
        else:
            self.drug_proj = nn.Sequential(
                nn.Linear(drug_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU()
            )
        # 细胞系需要从256降维到128
        self.cell_proj = nn.Sequential(
            nn.Linear(cell_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, drug_emb, cell_emb):
        """
        Args:
            drug_emb: [batch, drug_dim] 药物嵌入
            cell_emb: [batch, cell_dim] 细胞系嵌入
        Returns:
            drug_aligned: [batch, output_dim]
            cell_aligned: [batch, output_dim]
        """
        return self.drug_proj(drug_emb), self.cell_proj(cell_emb)


class BatchHypergraphBuilder:
    """为当前批次构建超图结构"""

    @staticmethod
    def build_hypergraph(batch_data, drug_emb_a, drug_emb_b, cell_emb):
        """
        构建批次级别的超图

        Args:
            batch_data: PyTorch Geometric Batch对象
            drug_emb_a: [batch, dim] 药物A嵌入
            drug_emb_b: [batch, dim] 药物B嵌入
            cell_emb: [batch, dim] 细胞系嵌入

        Returns:
            node_features: [N_unique_nodes, dim] 节点特征矩阵
            hyperedge_index: [2, N_edges] 超图邻接矩阵
            node_mapping: dict 节点ID到索引的映射
            batch_node_indices: dict 批次样本到节点索引的映射
        """
        device = drug_emb_a.device
        batch_size = len(batch_data.y)
        emb_dim = drug_emb_a.shape[1]

        # 1. 收集唯一的药物和细胞系ID
        drug_ids_a = batch_data.drug_a_id.cpu().numpy()
        drug_ids_b = batch_data.drug_b_id.cpu().numpy()
        cell_ids = batch_data.cell_id.cpu().numpy()

        # 2. 创建全局节点映射
        unique_drugs = np.unique(np.concatenate([drug_ids_a, drug_ids_b]))
        unique_cells = np.unique(cell_ids)

        node_mapping = {}
        node_idx = 0

        # 药物节点
        for drug_id in unique_drugs:
            node_mapping[('drug', int(drug_id))] = node_idx
            node_idx += 1

        # 细胞系节点
        for cell_id in unique_cells:
            node_mapping[('cell', int(cell_id))] = node_idx
            node_idx += 1

        N_nodes = len(node_mapping)

        # 3. 聚合节点特征（处理同一药物/细胞系在批次中多次出现）
        node_features = torch.zeros(N_nodes, emb_dim, device=device)
        node_counts = torch.zeros(N_nodes, device=device)

        # 记录每个批次样本对应的节点索引
        batch_node_indices = {
            'drug_a': torch.zeros(batch_size, dtype=torch.long, device=device),
            'drug_b': torch.zeros(batch_size, dtype=torch.long, device=device),
            'cell': torch.zeros(batch_size, dtype=torch.long, device=device)
        }

        for i in range(batch_size):
            # 药物A
            idx_a = node_mapping[('drug', int(drug_ids_a[i]))]
            node_features[idx_a] += drug_emb_a[i]
            node_counts[idx_a] += 1
            batch_node_indices['drug_a'][i] = idx_a

            # 药物B
            idx_b = node_mapping[('drug', int(drug_ids_b[i]))]
            node_features[idx_b] += drug_emb_b[i]
            node_counts[idx_b] += 1
            batch_node_indices['drug_b'][i] = idx_b

            # 细胞系
            idx_c = node_mapping[('cell', int(cell_ids[i]))]
            node_features[idx_c] += cell_emb[i]
            node_counts[idx_c] += 1
            batch_node_indices['cell'][i] = idx_c

        # 平均化（处理重复节点）
        node_features = node_features / node_counts.unsqueeze(1).clamp(min=1)

        # 4. 当前batch每行样本构建一条3节点超边：drug_a + drug_b + cell。
        hyperedge_list = []
        hyperedge_idx_list = []
        edge_count = 0

        for i in range(batch_size):
            idx_a = node_mapping[('drug', int(drug_ids_a[i]))]
            idx_b = node_mapping[('drug', int(drug_ids_b[i]))]
            idx_c = node_mapping[('cell', int(cell_ids[i]))]

            # 超边连接3个节点
            hyperedge_list.extend([idx_a, idx_b, idx_c])
            hyperedge_idx_list.extend([edge_count, edge_count, edge_count])
            edge_count += 1

        if len(hyperedge_list) == 0:
            # 防御空batch
            hyperedge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        else:
            hyperedge_index = torch.tensor(
                [hyperedge_list, hyperedge_idx_list],
                dtype=torch.long,
                device=device
            )

        return node_features, hyperedge_index, node_mapping, batch_node_indices


class HypergraphEncoder(nn.Module):
    """超图神经网络编码器"""
    def __init__(self, in_channels=128, hidden_channels=256, out_channels=128, dropout=0.2):
        super().__init__()

        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = HypergraphConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = HypergraphConv(hidden_channels, out_channels)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 用于无超边情况的线性变换
        self.linear_fallback = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, hyperedge_index):
        """
        Args:
            x: [N_nodes, in_channels] 节点特征
            hyperedge_index: [2, N_edges] 超图邻接矩阵
        Returns:
            x: [N_nodes, out_channels] 增强的节点嵌入
        """
        # 如果没有超边，使用线性变换作为fallback
        if hyperedge_index.shape[1] == 0:
            return self.linear_fallback(x)

        # 正常超图卷积
        x = self.conv1(x, hyperedge_index)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x, hyperedge_index)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv3(x, hyperedge_index)
        x = self.bn3(x)
        x = self.act(x)

        return x


class MeanHypergraphLayer(nn.Module):
    """无边函数的超图平均消息层：nodes -> hyperedges -> nodes。"""
    def __init__(self):
        super().__init__()

    @staticmethod
    def _mean_aggregate(values, index, dim_size):
        out = values.new_zeros(dim_size, values.size(-1))
        counts = values.new_zeros(dim_size, 1)
        out.index_add_(0, index, values)
        counts.index_add_(0, index, values.new_ones(values.size(0), 1))
        return out / counts.clamp(min=1)

    def forward(self, x, hyperedge_index):
        if hyperedge_index.shape[1] == 0:
            return x

        node_idx = hyperedge_index[0]
        edge_idx = hyperedge_index[1]
        num_edges = int(edge_idx.max().item()) + 1

        edge_msg = self._mean_aggregate(x[node_idx], edge_idx, num_edges)
        node_msg = self._mean_aggregate(edge_msg[edge_idx], node_idx, x.size(0))
        return node_msg


class MeanHypergraphEncoder(nn.Module):
    """纯平均超图编码器：超边取节点均值，节点取超边均值，ReLU进入下一层。"""
    def __init__(self, in_channels=128, hidden_channels=256, out_channels=128, dropout=0.2):
        super().__init__()

        self.input_proj = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)
        self.layers = nn.ModuleList([MeanHypergraphLayer() for _ in range(3)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(3)])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hyperedge_index):
        x = self.input_proj(x)
        for layer_idx, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x, hyperedge_index)
            x = norm(x)
            x = self.act(x)
            if layer_idx < len(self.layers) - 1:
                x = self.dropout(x)
        return x


class KANHypergraphLayer(nn.Module):
    """KAN版超图消息层：node -> hyperedge -> node。"""
    def __init__(self, in_channels, out_channels, dropout=0.2, use_kan=True, kan_type='fourier'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_kan = use_kan

        if use_kan:
            self.edge_func = KANLinear(in_channels, out_channels, kan_type=kan_type)
            self.node_func = KANLinear(out_channels, out_channels, kan_type=kan_type)
        else:
            self.edge_func = nn.Linear(in_channels, out_channels)
            self.node_func = nn.Linear(out_channels, out_channels)

        self.skip = nn.Identity() if in_channels == out_channels else nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _mean_aggregate(values, index, dim_size):
        out = values.new_zeros(dim_size, values.size(-1))
        counts = values.new_zeros(dim_size, 1)
        out.index_add_(0, index, values)
        counts.index_add_(0, index, values.new_ones(values.size(0), 1))
        return out / counts.clamp(min=1)

    def forward(self, x, hyperedge_index):
        if hyperedge_index.shape[1] == 0:
            return self.skip(x)

        node_idx = hyperedge_index[0]
        edge_idx = hyperedge_index[1]
        num_edges = int(edge_idx.max().item()) + 1

        edge_msg = self._mean_aggregate(x[node_idx], edge_idx, num_edges)
        edge_msg = self.edge_func(edge_msg)
        edge_msg = self.dropout(edge_msg)

        node_msg = self._mean_aggregate(edge_msg[edge_idx], node_idx, x.size(0))
        node_msg = self.node_func(node_msg)

        return self.skip(x) + node_msg


class KANHypergraphEncoder(nn.Module):
    """保持原始超图结构，只把超边消息函数替换成KAN。"""
    def __init__(self, in_channels=128, hidden_channels=256, out_channels=128,
                 dropout=0.2, use_kan=True, kan_type='fourier'):
        super().__init__()

        self.layer1 = KANHypergraphLayer(in_channels, hidden_channels, dropout, use_kan, kan_type)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.layer2 = KANHypergraphLayer(hidden_channels, hidden_channels, dropout, use_kan, kan_type)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.layer3 = KANHypergraphLayer(hidden_channels, out_channels, dropout, use_kan, kan_type)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hyperedge_index):
        x = self.layer1(x, hyperedge_index)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.layer2(x, hyperedge_index)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.layer3(x, hyperedge_index)
        x = self.bn3(x)
        x = self.act(x)

        return x


class KANAggregatedGraphBuilder:
    """为KAN聚合版构建普通图结构（将药物对聚合后与细胞构成边）"""

    def __init__(self, pair_edge_mode='shared_drug', pair_knn_k=5, pair_edge_seed=42):
        self.pair_edge_mode = pair_edge_mode
        self.pair_knn_k = int(pair_knn_k)
        self.pair_edge_seed = int(pair_edge_seed)

    def _stable_pair_seed(self, pair_id):
        a, b = pair_id
        return (self.pair_edge_seed + int(a) * 1000003 + int(b) * 9176) % (2 ** 32 - 1)

    def _add_shared_drug_edges(self, edge_set, unique_drug_pairs, node_mapping):
        for i, pair_i in enumerate(unique_drug_pairs):
            idx_i = node_mapping[('drug_pair', pair_i)]
            drugs_i = set(pair_i)
            for pair_j in unique_drug_pairs[i + 1:]:
                if drugs_i.intersection(pair_j):
                    idx_j = node_mapping[('drug_pair', pair_j)]
                    edge_set.add((idx_i, idx_j))
                    edge_set.add((idx_j, idx_i))

    def _add_molecular_knn_edges(self, edge_set, unique_drug_pairs, node_mapping, node_features):
        n_pairs = len(unique_drug_pairs)
        if n_pairs <= 1 or self.pair_knn_k <= 0:
            return

        pair_indices = torch.tensor(
            [node_mapping[('drug_pair', pair_id)] for pair_id in unique_drug_pairs],
            dtype=torch.long,
            device=node_features.device
        )
        pair_features = F.normalize(node_features[pair_indices], p=2, dim=-1)
        sim = pair_features @ pair_features.t()
        sim.fill_diagonal_(-float('inf'))

        k = min(self.pair_knn_k, n_pairs - 1)
        knn = torch.topk(sim, k=k, dim=1).indices.cpu().numpy()
        for i, nbrs in enumerate(knn):
            idx_i = int(pair_indices[i].item())
            for j in nbrs:
                idx_j = int(pair_indices[int(j)].item())
                edge_set.add((idx_i, idx_j))
                edge_set.add((idx_j, idx_i))

    def _add_random_knn_edges(self, edge_set, unique_drug_pairs, node_mapping):
        n_pairs = len(unique_drug_pairs)
        if n_pairs <= 1 or self.pair_knn_k <= 0:
            return

        k = min(self.pair_knn_k, n_pairs - 1)
        all_indices = np.arange(n_pairs)
        for i, pair_i in enumerate(unique_drug_pairs):
            rng = np.random.RandomState(self._stable_pair_seed(pair_i))
            candidates = all_indices[all_indices != i]
            chosen = rng.choice(candidates, size=k, replace=False)
            idx_i = node_mapping[('drug_pair', pair_i)]
            for j in chosen:
                idx_j = node_mapping[('drug_pair', unique_drug_pairs[int(j)])]
                edge_set.add((idx_i, idx_j))
                edge_set.add((idx_j, idx_i))

    def build_graph(self, batch_data, drug_emb_a, drug_emb_b, cell_emb):
        """
        构建KAN聚合版图结构

        Args:
            batch_data: PyTorch Geometric Batch对象
            drug_emb_a: [batch, dim] 药物A嵌入
            drug_emb_b: [batch, dim] 药物B嵌入
            cell_emb: [batch, dim] 细胞系嵌入

        Returns:
            node_features: [N_nodes, dim] 节点特征矩阵
            edge_index: [2, N_edges] 图邻接矩阵
            node_mapping: dict 节点ID到索引的映射
            batch_node_indices: dict 批次样本到节点索引的映射
        """
        device = drug_emb_a.device
        batch_size = len(batch_data.y)
        emb_dim = drug_emb_a.shape[1]

        # 1. 先将每个样本的两个drug嵌入平均，得到药物对嵌入
        drug_pair_emb = (drug_emb_a + drug_emb_b) / 2.0  # [batch, dim]

        # 2. 收集唯一的药物对和细胞系ID
        # 药物对ID使用 (drug_a_id, drug_b_id) 元组
        drug_ids_a = batch_data.drug_a_id.cpu().numpy()
        drug_ids_b = batch_data.drug_b_id.cpu().numpy()
        cell_ids = batch_data.cell_id.cpu().numpy()

        # 创建药物对ID（排序后的元组，确保(A,B)和(B,A)被视为同一对）
        drug_pair_ids = [tuple(sorted([int(drug_ids_a[i]), int(drug_ids_b[i])]))
                        for i in range(batch_size)]

        # 3. 创建全局节点映射
        unique_drug_pairs = sorted(set(drug_pair_ids))
        unique_cells = np.unique(cell_ids)

        node_mapping = {}
        node_idx = 0

        # 药物对节点
        for drug_pair_id in unique_drug_pairs:
            node_mapping[('drug_pair', drug_pair_id)] = node_idx
            node_idx += 1

        # 细胞系节点
        for cell_id in unique_cells:
            node_mapping[('cell', int(cell_id))] = node_idx
            node_idx += 1

        N_nodes = len(node_mapping)

        # 4. 聚合节点特征（处理同一药物对/细胞系在批次中多次出现）
        node_features = torch.zeros(N_nodes, emb_dim, device=device)
        node_counts = torch.zeros(N_nodes, device=device)

        # 记录每个批次样本对应的节点索引
        batch_node_indices = {
            'drug_pair': torch.zeros(batch_size, dtype=torch.long, device=device),
            'cell': torch.zeros(batch_size, dtype=torch.long, device=device)
        }

        for i in range(batch_size):
            # 药物对
            idx_pair = node_mapping[('drug_pair', drug_pair_ids[i])]
            node_features[idx_pair] += drug_pair_emb[i]
            node_counts[idx_pair] += 1
            batch_node_indices['drug_pair'][i] = idx_pair

            # 细胞系
            idx_c = node_mapping[('cell', int(cell_ids[i]))]
            node_features[idx_c] += cell_emb[i]
            node_counts[idx_c] += 1
            batch_node_indices['cell'][i] = idx_c

        # 平均化（处理重复节点）
        node_features = node_features / node_counts.unsqueeze(1).clamp(min=1)

        # 5. 当前batch内pair-cell边；pair-pair边由 pair_edge_mode 控制。
        edge_set = set()

        for i in range(batch_size):
            idx_pair = node_mapping[('drug_pair', drug_pair_ids[i])]
            idx_c = node_mapping[('cell', int(cell_ids[i]))]
            edge_set.add((idx_pair, idx_c))
            edge_set.add((idx_c, idx_pair))

        if self.pair_edge_mode in ['shared_drug', 'shared_plus_molecular_knn']:
            self._add_shared_drug_edges(edge_set, unique_drug_pairs, node_mapping)
        elif self.pair_edge_mode == 'none':
            pass

        if self.pair_edge_mode in ['molecular_knn', 'shared_plus_molecular_knn']:
            self._add_molecular_knn_edges(edge_set, unique_drug_pairs, node_mapping, node_features)
        elif self.pair_edge_mode == 'random_knn':
            self._add_random_knn_edges(edge_set, unique_drug_pairs, node_mapping)
        elif self.pair_edge_mode not in [
            'none',
            'shared_drug',
            'molecular_knn',
            'shared_plus_molecular_knn',
            'random_knn',
        ]:
            raise ValueError(f"Unknown pair_edge_mode: {self.pair_edge_mode}")

        if len(edge_set) == 0:
            # 防御空batch
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        else:
            edge_list = sorted(edge_set)
            edge_index = torch.tensor(
                edge_list,
                dtype=torch.long,
                device=device
            ).t().contiguous()

        return node_features, edge_index, node_mapping, batch_node_indices


class KANAggregatedGraphEncoder(nn.Module):
    """KAN聚合版图编码器，使用KAGCNConv"""
    def __init__(self, in_channels=128, hidden_channels=256, out_channels=128,
                 dropout=0.2, use_kan=True, kan_type='fourier'):
        super().__init__()

        self.use_kan = use_kan

        # 三层KAGCNConv
        self.conv1 = KAGCNConv(in_channels, hidden_channels, use_kan=use_kan, kan_type=kan_type)
        self.bn1 = nn.BatchNorm1d(hidden_channels)

        self.conv2 = KAGCNConv(hidden_channels, hidden_channels, use_kan=use_kan, kan_type=kan_type)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.conv3 = KAGCNConv(hidden_channels, out_channels, use_kan=use_kan, kan_type=kan_type)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 用于无边情况的线性变换
        self.linear_fallback = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, edge_index):
        """
        Args:
            x: [N_nodes, in_channels] 节点特征
            edge_index: [2, N_edges] 图邻接矩阵
        Returns:
            x: [N_nodes, out_channels] 增强的节点嵌入
        """
        # 如果没有边，使用线性变换作为fallback
        if edge_index.shape[1] == 0:
            return self.linear_fallback(x)

        # KAGCNConv: h' = h + KAN((h + mean(邻居)) / 2)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.act(x)

        return x


class HypergraphDecoder(nn.Module):
    """基于超图嵌入的协同作用预测器"""
    def __init__(self, in_channels=384, hidden1=256, hidden2=128, num_classes=2,
                 dropout=0.2, task='classification'):
        super().__init__()
        self.task = task

        self.fc1 = nn.Linear(in_channels, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)

        output_dim = 1 if task == 'regression' else num_classes
        self.fc3 = nn.Linear(hidden2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.uniform_(-1, 0)

    def forward(self, h_a, h_b, h_e):
        """
        Args:
            h_a: [batch, dim] 药物A的超图嵌入
            h_b: [batch, dim] 药物B的超图嵌入
            h_e: [batch, dim] 细胞系的超图嵌入
        Returns:
            classification: [batch, num_classes] log-softmax分数
            regression: [batch] raw regression score
        """
        x = torch.cat([h_a, h_b, h_e], dim=-1)  # [batch, 3*dim]

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.fc3(x)
        if self.task == 'regression':
            return x.squeeze(-1)
        return self.log_softmax(x)


class KANDecoder(nn.Module):
    """两层KAN融合头：用于原版DDoS/DeepDDS的concat+MLP直接对照。"""
    def __init__(self, in_channels=384, hidden1=256, hidden2=128, num_classes=2,
                 dropout=0.2, task='classification', kan_type='fourier'):
        super().__init__()
        self.task = task

        self.kan1 = KANLinear(in_channels, hidden1, kan_type=kan_type)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.kan2 = KANLinear(hidden1, hidden2, kan_type=kan_type)
        self.bn2 = nn.BatchNorm1d(hidden2)

        output_dim = 1 if task == 'regression' else num_classes
        self.out = nn.Linear(hidden2, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=-1)

        nn.init.xavier_normal_(self.out.weight.data)
        if self.out.bias is not None:
            self.out.bias.data.uniform_(-1, 0)

    def forward(self, h_a, h_b, h_e):
        x = torch.cat([h_a, h_b, h_e], dim=-1)

        x = self.kan1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.kan2(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.out(x)
        if self.task == 'regression':
            return x.squeeze(-1)
        return self.log_softmax(x)


class GeneGraphExpressionEncoder(nn.Module):
    """Encode cell-line expression with a fixed gene-gene graph."""
    def __init__(self, num_genes, hidden_channels=128, out_channels=256,
                 num_layers=2, dropout=0.2, use_kan=False, kan_type='fourier',
                 edge_index=None):
        super().__init__()
        self.num_genes = int(num_genes)
        self.hidden_channels = hidden_channels

        if edge_index is None:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        self.register_buffer('edge_index', edge_index.long())

        self.value_encoder = nn.Linear(1, hidden_channels)
        self.gene_embedding = nn.Embedding(self.num_genes, hidden_channels)
        self.convs = nn.ModuleList([
            KAGCNConv(hidden_channels, hidden_channels, use_kan=use_kan, kan_type=kan_type)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        ])
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.dropout = nn.Dropout(dropout)

    def _batched_edge_index(self, batch_size, device):
        if self.edge_index.numel() == 0:
            return self.edge_index.to(device)

        num_edges = self.edge_index.size(1)
        offsets = (
            torch.arange(batch_size, device=device)
            .repeat_interleave(num_edges) * self.num_genes
        )
        return self.edge_index.to(device).repeat(1, batch_size) + offsets.unsqueeze(0)

    def forward(self, expression):
        batch_size, num_genes = expression.shape
        if num_genes != self.num_genes:
            raise ValueError(f"Expected {self.num_genes} genes, got {num_genes}.")

        gene_ids = torch.arange(num_genes, device=expression.device)
        gene_emb = self.gene_embedding(gene_ids).unsqueeze(0).expand(batch_size, -1, -1)
        value_emb = self.value_encoder(expression.reshape(-1, 1)).view(
            batch_size, num_genes, self.hidden_channels
        )
        h = (value_emb + gene_emb).reshape(batch_size * num_genes, self.hidden_channels)

        edge_index = self._batched_edge_index(batch_size, expression.device)
        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            if layer_idx < len(self.convs) - 1:
                h = self.dropout(h)

        h_graph = h.view(batch_size, num_genes, self.hidden_channels).mean(dim=1)
        return self.out_proj(h_graph)


class DeepDDS_Hypergraph(nn.Module):
    """
    集成超图的DeepDDS模型（支持多种模式）

    模式选择：
    - 'mlp': 不使用图结构，直接MLP
    - 'kan_mlp': 不使用图结构，直接两层KAN融合头
    - 'hypergraph': 原始超图（3节点超边：drug_a + drug_b + cell）
    - 'mean_hypergraph': 原始超图结构 + 无边函数mean-mean-ReLU消息传递
    - 'kan_hypergraph': 原始超图结构 + KAN超边消息函数
    - 'kan_aggregated': KAN聚合版（drug_pair-cell + shared-drug pair-pair）
    """
    def __init__(self,
                 num_features_xd=9,
                 gat_output_dim=128,
                 expression_input_size=926,
                 exp_H1=8192,
                 exp_H2=4096,
                 unified_dim=128,
                 hypergraph_hidden=256,
                 decoder_hidden1=256,
                 decoder_hidden2=128,
                 num_classes=2,
                 dropout=0.2,
                 num_attn_heads=10,
                 hypergraph_mode='hypergraph',
                 use_kan=True,
                 kan_type='fourier',
                 use_drug_kan=True,
                 use_drug_readout_kan=False,
                 gnn_type='gcn',
                 num_layer=5,
                 paper_node_feature_dim=None,
                 paper_edge_feature_dim=None,
                 drug_jk='last',
                 graph_pooling='mean',
                 decoder_type='mlp',
                 pair_edge_mode='shared_drug',
                 pair_knn_k=5,
                 pair_edge_seed=42,
                 cell_encoder_type='mlp',
                 cell_graph_edge_index=None,
                 cell_graph_hidden=128,
                 cell_graph_layers=2,
                 cell_graph_use_kan=False,
                 task='classification'):
        super().__init__()

        self.hypergraph_mode = hypergraph_mode  # 保存模式状态
        self.use_drug_kan = use_drug_kan
        self.task = task
        self.decoder_type = 'kan' if hypergraph_mode == 'kan_mlp' else decoder_type
        self.pair_edge_mode = pair_edge_mode
        self.cell_encoder_type = cell_encoder_type
        self.drug_features_are_raw_float = (
            gnn_type in ['paper_kagcn', 'ka_gcn_paper', 'kagcn_paper']
            and paper_node_feature_dim is not None
            and paper_edge_feature_dim is not None
        )

        # Stage 1: 药物编码器
        self.drug_encoder_uses_atom_encoder = gnn_type != 'gatnet'
        if self.drug_encoder_uses_atom_encoder:
            self.drug_encoder = KAGNN(
                gnn_type=gnn_type,
                num_layer=num_layer,
                emb_dim=gat_output_dim,
                drop_ratio=dropout,
                JK=drug_jk,
                graph_pooling=graph_pooling,
                virtual_node=False,
                with_edge_attr=False,
                use_kan=use_drug_kan,
                kan_type=kan_type,
                use_readout_kan=use_drug_readout_kan,
                paper_node_feature_dim=paper_node_feature_dim,
                paper_edge_feature_dim=paper_edge_feature_dim
            )
        else:
            self.drug_encoder = GATNet(
                num_features_xd=num_features_xd,
                n_output=num_classes,
                output_dim=gat_output_dim,
                dropout=dropout,
                heads=num_attn_heads
            )

        if self.cell_encoder_type == 'gene_graph':
            self.expression_encoder = GeneGraphExpressionEncoder(
                num_genes=expression_input_size,
                hidden_channels=cell_graph_hidden,
                out_channels=2*unified_dim,
                num_layers=cell_graph_layers,
                dropout=dropout,
                use_kan=cell_graph_use_kan,
                kan_type=kan_type,
                edge_index=cell_graph_edge_index
            )
        else:
            self.expression_encoder = ExpressionNN(
                D_in=expression_input_size,
                H1=exp_H1,
                H2=exp_H2,
                D_out=2*unified_dim,  # 输出256维
                drop=dropout
            )

        # Stage 2: 嵌入对齐
        self.embedding_aligner = EmbeddingAligner(
            drug_dim=gat_output_dim,
            cell_dim=2*unified_dim,
            output_dim=unified_dim
        )

        # Stage 3: 图编码器（根据模式选择）
        if self.hypergraph_mode == 'hypergraph':
            # 原始超图模式
            self.hypergraph_encoder = HypergraphEncoder(
                in_channels=unified_dim,
                hidden_channels=hypergraph_hidden,
                out_channels=unified_dim,
                dropout=dropout
            )
            self.hypergraph_builder = BatchHypergraphBuilder()
        elif self.hypergraph_mode == 'mean_hypergraph':
            self.hypergraph_encoder = MeanHypergraphEncoder(
                in_channels=unified_dim,
                hidden_channels=hypergraph_hidden,
                out_channels=unified_dim,
                dropout=dropout
            )
            self.hypergraph_builder = BatchHypergraphBuilder()
        elif self.hypergraph_mode == 'kan_hypergraph':
            self.hypergraph_encoder = KANHypergraphEncoder(
                in_channels=unified_dim,
                hidden_channels=hypergraph_hidden,
                out_channels=unified_dim,
                dropout=dropout,
                use_kan=use_kan,
                kan_type=kan_type
            )
            self.hypergraph_builder = BatchHypergraphBuilder()
        elif self.hypergraph_mode == 'kan_aggregated':
            # KAN聚合版图模式
            self.kan_graph_encoder = KANAggregatedGraphEncoder(
                in_channels=unified_dim,
                hidden_channels=hypergraph_hidden,
                out_channels=unified_dim,
                dropout=dropout,
                use_kan=use_kan,
                kan_type=kan_type
            )
            self.kan_graph_builder = KANAggregatedGraphBuilder(
                pair_edge_mode=pair_edge_mode,
                pair_knn_k=pair_knn_k,
                pair_edge_seed=pair_edge_seed,
            )
        # else: mlp模式不需要图编码器

        # Stage 4: 解码器
        decoder_kwargs = {
            'in_channels': 3*unified_dim,
            'hidden1': decoder_hidden1,
            'hidden2': decoder_hidden2,
            'num_classes': num_classes,
            'dropout': dropout,
            'task': task,
        }
        if self.decoder_type == 'kan':
            self.decoder = KANDecoder(**decoder_kwargs, kan_type=kan_type)
        else:
            self.decoder = HypergraphDecoder(**decoder_kwargs)

        self.unified_dim = unified_dim

    def _encode_drug(self, x, edge_index, edge_attr, batch):
        if self.drug_encoder_uses_atom_encoder:
            if self.drug_features_are_raw_float:
                return self.drug_encoder(x.float(), edge_index, edge_attr, batch)
            return self.drug_encoder(x.long(), edge_index, edge_attr, batch)
        return self.drug_encoder(x.type(torch.float32), edge_index, edge_attr, batch)

    def _encode_modalities(self, batch):
        h_a_raw = self._encode_drug(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
        h_b_raw = self._encode_drug(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)
        h_e_raw = self.expression_encoder(batch.expression)

        h_a_aligned, h_e_aligned = self.embedding_aligner(h_a_raw, h_e_raw)
        h_b_aligned, _ = self.embedding_aligner(h_b_raw, h_e_raw)
        return h_a_aligned, h_b_aligned, h_e_aligned

    def forward(self, batch):
        """
        Args:
            batch: PyTorch Geometric Batch对象，包含：
                - x_a, edge_index_a, edge_attr_a, x_a_batch: 药物A图数据
                - x_b, edge_index_b, edge_attr_b, x_b_batch: 药物B图数据
                - expression: 细胞系基因表达
                - drug_a_id, drug_b_id, cell_id: 节点ID（图模式需要）
                - y: 标签
        Returns:
            log_probs: [batch, num_classes] 预测分数
        """
        # Stage 1-2: 仅编码当前query batch。
        h_a_aligned, h_b_aligned, h_e_aligned = self._encode_modalities(batch)
        # h_a_aligned, h_b_aligned, h_e_aligned: [batch, unified_dim]

        if self.hypergraph_mode in ['hypergraph', 'mean_hypergraph', 'kan_hypergraph']:
            # ===== 原始超图模式 =====
            # Stage 3: 构建超图（3节点超边：drug_a + drug_b + cell）
            node_features, hyperedge_index, node_mapping, batch_node_indices = \
                self.hypergraph_builder.build_hypergraph(
                    batch, h_a_aligned, h_b_aligned, h_e_aligned
                )

            # Stage 4: 超图卷积
            node_embeddings = self.hypergraph_encoder(
                node_features, hyperedge_index
            )

            # Stage 5: 提取对应的节点嵌入
            h_a_enhanced = node_embeddings[batch_node_indices['drug_a']]
            h_b_enhanced = node_embeddings[batch_node_indices['drug_b']]
            h_e_enhanced = node_embeddings[batch_node_indices['cell']]

        elif self.hypergraph_mode == 'kan_aggregated':
            # ===== KAN聚合版图模式 =====
            # Stage 3: 构建普通图（2节点边：drug_pair + cell）
            node_features, edge_index, node_mapping, batch_node_indices = \
                self.kan_graph_builder.build_graph(
                    batch, h_a_aligned, h_b_aligned, h_e_aligned
                )

            # Stage 4: KAGCNConv图卷积：h' = h + KAN((h + mean(邻居)) / 2)
            node_embeddings = self.kan_graph_encoder(
                node_features, edge_index
            )

            # Stage 5: 提取对应的节点嵌入
            # 注意：KAN聚合版中，drug_pair已经是平均后的结果
            # 但decoder期望3个输入，所以我们需要将drug_pair拆分回两个drug
            drug_pair_emb = node_embeddings[batch_node_indices['drug_pair']]
            h_e_enhanced = node_embeddings[batch_node_indices['cell']]

            # 将drug_pair嵌入复制给两个drug（因为它们已经被聚合了）
            h_a_enhanced = drug_pair_emb
            h_b_enhanced = drug_pair_emb

        else:  # mlp模式
            # ===== 简单MLP模式（原始DeepDDS方式）=====
            # 直接使用对齐后的嵌入，不经过图结构
            h_a_enhanced = h_a_aligned
            h_b_enhanced = h_b_aligned
            h_e_enhanced = h_e_aligned

        # Stage 6: 预测（三种模式共用）
        log_probs = self.decoder(h_a_enhanced, h_b_enhanced, h_e_enhanced)

        return log_probs


class DeepDDS_Hypergraph_WithReconstruction(DeepDDS_Hypergraph):
    """带重构损失的超图DeepDDS模型（仅在hypergraph模式时有效）"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 重构权重矩阵（仅在使用超图时有意义）
        if self.hypergraph_mode in ['hypergraph', 'mean_hypergraph', 'kan_hypergraph']:
            unified_dim = kwargs.get('unified_dim', 128)
            self.drug_rec_weight = nn.Parameter(torch.randn(unified_dim, unified_dim))
            self.cell_rec_weight = nn.Parameter(torch.randn(unified_dim, unified_dim))

            nn.init.xavier_uniform_(self.drug_rec_weight)
            nn.init.xavier_uniform_(self.cell_rec_weight)

    def forward(self, batch):
        """
        Returns:
            log_probs: [batch, num_classes] 预测分数
            rec_drug: [N_drugs, N_drugs] 重构的药物相似度矩阵（仅hypergraph模式）
            rec_cell: [N_cells, N_cells] 重构的细胞系相似度矩阵（仅hypergraph模式）
        """
        device = batch.x_a.device

        # Stage 1-2: 仅编码当前query batch。
        h_a_aligned, h_b_aligned, h_e_aligned = self._encode_modalities(batch)

        if self.hypergraph_mode in ['hypergraph', 'mean_hypergraph', 'kan_hypergraph']:
            # ===== 超图模式 =====
            # Stage 3-4: 超图构建和卷积
            node_features, hyperedge_index, node_mapping, batch_node_indices = \
                self.hypergraph_builder.build_hypergraph(
                    batch, h_a_aligned, h_b_aligned, h_e_aligned
                )
            node_embeddings = self.hypergraph_encoder(node_features, hyperedge_index)

            # Stage 5: 提取嵌入
            h_a_enhanced = node_embeddings[batch_node_indices['drug_a']]
            h_b_enhanced = node_embeddings[batch_node_indices['drug_b']]
            h_e_enhanced = node_embeddings[batch_node_indices['cell']]

            # Stage 6: 预测
            log_probs = self.decoder(h_a_enhanced, h_b_enhanced, h_e_enhanced)

            # Stage 7: 重构相似度矩阵
            # 分离药物和细胞系节点
            drug_mask = torch.tensor([k[0] == 'drug' for k in node_mapping.keys()],
                                     dtype=torch.bool, device=device)
            cell_mask = ~drug_mask

            drug_embeddings = node_embeddings[drug_mask]
            cell_embeddings = node_embeddings[cell_mask]

            # 重构药物相似度
            rec_drug = torch.sigmoid(
                torch.mm(torch.mm(drug_embeddings, self.drug_rec_weight), drug_embeddings.t())
            )

            # 重构细胞系相似度
            rec_cell = torch.sigmoid(
                torch.mm(torch.mm(cell_embeddings, self.cell_rec_weight), cell_embeddings.t())
            )

            return log_probs, rec_drug, rec_cell
        else:
            # ===== 其他模式（mlp或kan_aggregated）=====
            # 不使用重构损失
            if self.hypergraph_mode == 'kan_aggregated':
                node_features, edge_index, node_mapping, batch_node_indices = \
                    self.kan_graph_builder.build_graph(
                        batch, h_a_aligned, h_b_aligned, h_e_aligned
                    )
                node_embeddings = self.kan_graph_encoder(node_features, edge_index)

                drug_pair_emb = node_embeddings[batch_node_indices['drug_pair']]
                h_e_enhanced = node_embeddings[batch_node_indices['cell']]
                h_a_enhanced = drug_pair_emb
                h_b_enhanced = drug_pair_emb
            else:  # mlp模式
                h_a_enhanced = h_a_aligned
                h_b_enhanced = h_b_aligned
                h_e_enhanced = h_e_aligned

            log_probs = self.decoder(h_a_enhanced, h_b_enhanced, h_e_enhanced)

            # 返回None作为重构矩阵（表示不使用重构损失）
            return log_probs, None, None
