"""
KAN-enhanced GNN (KAGNN) 主模型
完整的drug端KAGNN实现，支持GNN/KAGNN切换
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention

from .kagnn_node import KAGNN_node, KAGNN_node_Virtualnode
from .model_attn_siamese import FeatureEmbAttention as GNNLayerEmbAttention
from .kan_layers import KANLinear


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
                 use_kan=False, kan_type='fourier', use_readout_kan=False):
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

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        node_jk = "multilayer" if JK == "concat_kan" else JK

        # GNN节点嵌入生成器
        if virtual_node:
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
        if self.JK == "multilayer":
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

        if self.JK == "concat_kan":
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
        if self.with_edge_attr:
            h_node = self.gnn_node(x, edge_index, edge_attr)
        else:
            h_node = self.gnn_node(x, edge_index, None)

        # 图级池化
        if self.JK in ["multilayer", "concat_kan"]:
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
            h_graph = self.readout_norm(self.readout_kan(h_graph))

        return h_graph
