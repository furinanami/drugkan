"""
DeepDDS with Hypergraph Integration
集成超图神经网络的DeepDDS模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
import numpy as np

try:
    from .model_gnn_ogb import GATNet, ExpressionNN
except ImportError:
    from model_gnn_ogb import GATNet, ExpressionNN


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

        # 4. 构建超边（只使用正样本）
        hyperedge_list = []
        hyperedge_idx_list = []
        edge_count = 0

        for i in range(batch_size):
            if batch_data.y[i].item() == 1:  # 只为正样本创建超边
                idx_a = batch_node_indices['drug_a'][i].item()
                idx_b = batch_node_indices['drug_b'][i].item()
                idx_c = batch_node_indices['cell'][i].item()

                # 超边连接3个节点
                hyperedge_list.extend([idx_a, idx_b, idx_c])
                hyperedge_idx_list.extend([edge_count, edge_count, edge_count])
                edge_count += 1

        if len(hyperedge_list) == 0:
            # 如果批次中没有正样本，创建空超图
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


class HypergraphDecoder(nn.Module):
    """基于超图嵌入的协同作用预测器"""
    def __init__(self, in_channels=384, hidden1=256, hidden2=128, num_classes=2, dropout=0.2):
        super().__init__()

        self.fc1 = nn.Linear(in_channels, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)

        self.fc3 = nn.Linear(hidden2, num_classes)

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
            log_probs: [batch, num_classes] log-softmax分数
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
        return self.log_softmax(x)


class DeepDDS_Hypergraph(nn.Module):
    """集成超图的DeepDDS模型（支持开关控制是否使用超图）"""
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
                 use_hypergraph=True):  # 新增开关参数
        super().__init__()

        self.use_hypergraph = use_hypergraph  # 保存开关状态

        # Stage 1: 原有编码器
        self.gat_encoder = GATNet(
            num_features_xd=num_features_xd,
            n_output=num_classes,
            output_dim=gat_output_dim,
            dropout=dropout,
            heads=num_attn_heads
        )

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

        # Stage 3: 超图编码器（仅在use_hypergraph=True时使用）
        if self.use_hypergraph:
            self.hypergraph_encoder = HypergraphEncoder(
                in_channels=unified_dim,
                hidden_channels=hypergraph_hidden,
                out_channels=unified_dim,
                dropout=dropout
            )
            self.hypergraph_builder = BatchHypergraphBuilder()

        # Stage 4: 解码器
        self.decoder = HypergraphDecoder(
            in_channels=3*unified_dim,
            hidden1=decoder_hidden1,
            hidden2=decoder_hidden2,
            num_classes=num_classes,
            dropout=dropout
        )

        self.unified_dim = unified_dim

    def forward(self, batch):
        """
        Args:
            batch: PyTorch Geometric Batch对象，包含：
                - x_a, edge_index_a, edge_attr_a, x_a_batch: 药物A图数据
                - x_b, edge_index_b, edge_attr_b, x_b_batch: 药物B图数据
                - expression: 细胞系基因表达
                - drug_a_id, drug_b_id, cell_id: 节点ID（仅use_hypergraph=True时需要）
                - y: 标签
        Returns:
            log_probs: [batch, num_classes] 预测分数
        """
        device = batch.x_a.device
        batch_size = len(batch.y)

        # Stage 1: 编码原始特征
        h_a_raw = self.gat_encoder(
            batch.x_a.type(torch.float32),
            batch.edge_index_a,
            batch.edge_attr_a,
            batch.x_a_batch
        )  # [batch, 128]

        h_b_raw = self.gat_encoder(
            batch.x_b.type(torch.float32),
            batch.edge_index_b,
            batch.edge_attr_b,
            batch.x_b_batch
        )  # [batch, 128]

        h_e_raw = self.expression_encoder(batch.expression)  # [batch, 256]

        # Stage 2: 对齐嵌入维度
        h_a_aligned, h_e_aligned = self.embedding_aligner(h_a_raw, h_e_raw)
        h_b_aligned, _ = self.embedding_aligner(h_b_raw, h_e_raw)
        # h_a_aligned, h_b_aligned, h_e_aligned: [batch, 256]

        if self.use_hypergraph:
            # ===== 超图模式 =====
            # Stage 3: 构建超图
            node_features, hyperedge_index, node_mapping, batch_node_indices = \
                self.hypergraph_builder.build_hypergraph(
                    batch, h_a_aligned, h_b_aligned, h_e_aligned
                )
            # node_features: [N_nodes, 256]
            # hyperedge_index: [2, N_edges]

            # Stage 4: 超图卷积
            node_embeddings = self.hypergraph_encoder(
                node_features, hyperedge_index
            )  # [N_nodes, 256]

            # Stage 5: 提取对应的节点嵌入
            h_a_enhanced = node_embeddings[batch_node_indices['drug_a']]
            h_b_enhanced = node_embeddings[batch_node_indices['drug_b']]
            h_e_enhanced = node_embeddings[batch_node_indices['cell']]
        else:
            # ===== 简单MLP模式（原始DeepDDS方式）=====
            # 直接使用对齐后的嵌入，不经过超图
            h_a_enhanced = h_a_aligned
            h_b_enhanced = h_b_aligned
            h_e_enhanced = h_e_aligned

        # Stage 6: 预测（两种模式共用）
        log_probs = self.decoder(h_a_enhanced, h_b_enhanced, h_e_enhanced)

        return log_probs


class DeepDDS_Hypergraph_WithReconstruction(DeepDDS_Hypergraph):
    """带重构损失的超图DeepDDS模型（仅在use_hypergraph=True时有效）"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 重构权重矩阵（仅在使用超图时有意义）
        if self.use_hypergraph:
            unified_dim = kwargs.get('unified_dim', 256)
            self.drug_rec_weight = nn.Parameter(torch.randn(unified_dim, unified_dim))
            self.cell_rec_weight = nn.Parameter(torch.randn(unified_dim, unified_dim))

            nn.init.xavier_uniform_(self.drug_rec_weight)
            nn.init.xavier_uniform_(self.cell_rec_weight)

    def forward(self, batch):
        """
        Returns:
            log_probs: [batch, num_classes] 预测分数
            rec_drug: [N_drugs, N_drugs] 重构的药物相似度矩阵（仅use_hypergraph=True）
            rec_cell: [N_cells, N_cells] 重构的细胞系相似度矩阵（仅use_hypergraph=True）
        """
        device = batch.x_a.device
        batch_size = len(batch.y)

        # Stage 1-2: 编码和对齐
        h_a_raw = self.gat_encoder(
            batch.x_a.type(torch.float32),
            batch.edge_index_a,
            batch.edge_attr_a,
            batch.x_a_batch
        )
        h_b_raw = self.gat_encoder(
            batch.x_b.type(torch.float32),
            batch.edge_index_b,
            batch.edge_attr_b,
            batch.x_b_batch
        )
        h_e_raw = self.expression_encoder(batch.expression)

        h_a_aligned, h_e_aligned = self.embedding_aligner(h_a_raw, h_e_raw)
        h_b_aligned, _ = self.embedding_aligner(h_b_raw, h_e_raw)

        if self.use_hypergraph:
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
            # ===== 简单MLP模式 =====
            # 不使用超图时，不计算重构损失
            h_a_enhanced = h_a_aligned
            h_b_enhanced = h_b_aligned
            h_e_enhanced = h_e_aligned

            log_probs = self.decoder(h_a_enhanced, h_b_enhanced, h_e_enhanced)

            # 返回None作为重构矩阵（表示不使用重构损失）
            return log_probs, None, None
