"""
KAN-enhanced GNN (KAGNN) 主模型
完整的drug端KAGNN实现，支持GNN/KAGNN切换
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from .kagnn_node import KAGNN_node, KAGNN_node_Virtualnode
from .model_attn_siamese import FeatureEmbAttention as GNNLayerEmbAttention
from .kan_layers import KANLinear
from .kagnn_conv import KAGCNConv


class PaperKAGCNNodeEncoder(nn.Module):
    """
    KA-GCN drug node encoder following the paper equations.

    The paper uses KAN in three places for KA-GCN:
    1. node initialization: KAN(atom_feature || mean(incident_bond_feature))
    2. message passing: h_v + KAN(h_v || mean(h_u, u in N(v)))
    3. readout: KAN(mean_v h_v)

    In this project molecular edges are RDKit covalent bonds only; the existing
    OGB atom/bond encoders convert categorical atom and bond descriptors to
    dense features before the KAN modules.
    """
    def __init__(self, num_layer, emb_dim, drop_ratio=0.5,
                 use_kan=True, kan_type='fourier',
                 node_feature_dim=None, edge_feature_dim=None):
        super().__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.use_kan = use_kan
        self.raw_feature_mode = node_feature_dim is not None and edge_feature_dim is not None

        if self.raw_feature_mode:
            self.atom_encoder = None
            self.bond_encoder = None
            init_in = int(node_feature_dim) + int(edge_feature_dim)
            self.edge_feature_dim = int(edge_feature_dim)
        else:
            self.atom_encoder = AtomEncoder(emb_dim)
            self.bond_encoder = BondEncoder(emb_dim=emb_dim)
            init_in = 2 * emb_dim
            self.edge_feature_dim = emb_dim

        if use_kan:
            self.node_init = KANLinear(init_in, emb_dim, kan_type=kan_type)
        else:
            self.node_init = nn.Linear(init_in, emb_dim)

        self.convs = nn.ModuleList([
            KAGCNConv(emb_dim, emb_dim, use_kan=use_kan, kan_type=kan_type)
            for _ in range(num_layer)
        ])

    def _mean_incident_bonds(self, edge_index, edge_attr, num_nodes):
        if edge_attr is None or edge_attr.numel() == 0 or edge_index.numel() == 0:
            return edge_index.new_zeros((num_nodes, self.edge_feature_dim), dtype=torch.float32)

        if self.raw_feature_mode:
            edge_emb = edge_attr.float()
        else:
            edge_emb = self.bond_encoder(edge_attr.long())
        dst = edge_index[1]
        out = edge_emb.new_zeros(num_nodes, edge_emb.size(-1))
        counts = edge_emb.new_zeros(num_nodes, 1)
        out.index_add_(0, dst, edge_emb)
        counts.index_add_(0, dst, edge_emb.new_ones(edge_emb.size(0), 1))
        return out / counts.clamp(min=1)

    def forward(self, x, edge_index, edge_attr):
        if self.raw_feature_mode:
            atom_emb = x.float()
        else:
            atom_emb = self.atom_encoder(x.long())
        bond_mean = self._mean_incident_bonds(edge_index, edge_attr, atom_emb.size(0))
        h = self.node_init(torch.cat([atom_emb, bond_mean], dim=-1))

        for layer_idx, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if layer_idx < len(self.convs) - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)

        return h


class KAGNN(nn.Module):
    """
    KAN-enhanced Graph Neural Network for Drug Representation

    核心设计理念：
    1. 保持与原始GNN完全相同的接口和整体架构
    2. 通过use_kan开关无缝切换GNN/KAGNN模式
    3. 通过kan_type选择KAN实现（fourier/bspline）

    架构说明：
    - 输入：分子图（节点特征、边索引、边特征）
    - GNN层：多层消息传递，提取节点嵌入
    - 池化层：聚合节点嵌入为图级表示
    - 输出：固定维度的drug嵌入向量

    KAN增强的优势：
    1. 更强的非线性表达能力：KAN可以学习任意复杂的激活函数
    2. 自适应特征变换：根据数据自动学习最优的特征映射
    3. 更好的泛化性能：理论上KAN具有更强的函数逼近能力

    Args:
        num_layer: GNN层数（默认5）
        emb_dim: 嵌入维度（默认300）
        gnn_type: GNN类型 ('gat', 'gatv2', 'gcn', 'gin')
        virtual_node: 是否使用虚拟节点
        residual: 是否使用残差连接
        drop_ratio: Dropout比率
        JK: Jumping Knowledge模式
        graph_pooling: 图池化方式 ('mean', 'sum', 'max', 'attention')
        with_edge_attr: 是否使用边特征
        use_kan: 是否使用KAN增强（核心开关）
        kan_type: KAN类型 ('fourier' 或 'bspline'，默认'fourier')
    """
    def __init__(self, num_layer=5, emb_dim=300, gnn_type='gat',
                 virtual_node=True, residual=False, drop_ratio=0.5,
                 JK="last", graph_pooling="mean", with_edge_attr=False,
                 use_kan=False, kan_type='fourier', use_readout_kan=False,
                 paper_node_feature_dim=None, paper_edge_feature_dim=None):
        super(KAGNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.graph_pooling = graph_pooling
        self.with_edge_attr = with_edge_attr
        self.use_kan = use_kan
        self.kan_type = kan_type
        self.use_readout_kan = use_readout_kan
        self.paper_kagcn = gnn_type in ['paper_kagcn', 'ka_gcn_paper', 'kagcn_paper']

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        node_jk = "multilayer" if JK == "concat_kan" else JK

        # GNN节点嵌入生成器
        if self.paper_kagcn:
            self.gnn_node = PaperKAGCNNodeEncoder(
                num_layer, emb_dim, drop_ratio=drop_ratio,
                use_kan=use_kan, kan_type=kan_type,
                node_feature_dim=paper_node_feature_dim,
                edge_feature_dim=paper_edge_feature_dim
            )
            self.with_edge_attr = True
            self.use_readout_kan = use_kan
        elif virtual_node:
            self.gnn_node = KAGNN_node_Virtualnode(
                num_layer, emb_dim, JK=node_jk, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type,
                use_kan=use_kan, kan_type=kan_type
            )
        else:
            self.gnn_node = KAGNN_node(
                num_layer, emb_dim, JK=node_jk, drop_ratio=drop_ratio,
                residual=residual, gnn_type=gnn_type,
                with_edge_attr=self.with_edge_attr,
                use_kan=use_kan, kan_type=kan_type
            )

        # 层级池化注意力只用于 multilayer JK；concat_kan 直接拼接各层图表示后过 KAN。
        if self.JK == "multilayer" and not self.paper_kagcn:
            self.layer_pooling = GNNLayerEmbAttention(emb_dim)

        # 图级池化函数
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(emb_dim, 2 * emb_dim),
                    nn.BatchNorm1d(2 * emb_dim),
                    nn.ReLU(),
                    nn.Linear(2 * emb_dim, 1)
                )
            )
        else:
            raise ValueError("Invalid graph pooling type.")

        if self.use_readout_kan:
            self.readout_kan = KANLinear(emb_dim, emb_dim, kan_type=kan_type)
            self.readout_norm = nn.LayerNorm(emb_dim)

        if self.JK == "concat_kan" and not self.paper_kagcn:
            self.layer_concat_kan = KANLinear(num_layer * emb_dim, emb_dim, kan_type=kan_type)
            self.layer_concat_norm = nn.LayerNorm(emb_dim)

        # 打印模型配置
        mode_str = "KAN-enhanced" if use_kan else "Standard"
        kan_str = f" ({kan_type})" if use_kan else ""
        print(f"Initialized {mode_str} GNN{kan_str}: {gnn_type}, "
              f"{num_layer} layers, emb_dim={emb_dim}")

    def forward(self, x, edge_index, edge_attr, batch):
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, num_node_features]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, num_edge_features]
            batch: 批次索引 [num_nodes]

        Returns:
            h_graph: 图级嵌入 [batch_size, emb_dim]
        """
        # 生成节点嵌入
        if self.paper_kagcn:
            h_node = self.gnn_node(x, edge_index, edge_attr)
        elif self.with_edge_attr:
            h_node = self.gnn_node(x, edge_index, edge_attr)
        else:
            h_node = self.gnn_node(x, edge_index, None)

        # 图级池化
        if (not self.paper_kagcn) and self.JK in ["multilayer", "concat_kan"]:
            if self.JK == "concat_kan":
                h_node = h_node[1:]

            # 多层Jumping Knowledge：对每层输出分别池化
            h_graphs = [self.pool(h, batch) for h in h_node]

            # 拼接所有层的图表示
            h_graph_cat = torch.cat(h_graphs, dim=1)

            if self.JK == "concat_kan":
                h_graph = self.layer_concat_norm(self.layer_concat_kan(h_graph_cat))
            else:
                # 重塑为 [batch_size, num_layers, emb_dim]
                h_graph_t = h_graph_cat.reshape(
                    h_graph_cat.shape[0],
                    len(h_graphs),
                    h_graph_cat.shape[1] // len(h_graphs)
                )

                # 使用注意力机制加权聚合不同层的表示
                h_graph, layer_weights = self.layer_pooling(h_graph_t)
        else:
            # 单层输出直接池化
            h_graph = self.pool(h_node, batch)

        if self.use_readout_kan:
            h_graph = self.readout_kan(h_graph)
            if not self.paper_kagcn:
                h_graph = self.readout_norm(h_graph)

        return h_graph
