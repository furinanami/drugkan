"""
KAN-enhanced GNN Convolution Layers
将KAN集成到GNN的消息传递机制中，提升对复杂非线性交互的表示能力
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, degree, softmax
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from .kan_layers import KANLinear


class KAGATConv(MessagePassing):
    """
    KAN-enhanced Graph Attention Network (KAGAT)

    核心思想：
    1. 原始GAT使用固定的LeakyReLU激活函数计算注意力权重
    2. KAGAT使用KAN替代固定激活，让模型学习最优的注意力计算方式
    3. 在消息聚合后的变换中也使用KAN，增强特征表达能力

    数学原理：
    原始GAT: α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    KAGAT:   α_ij = softmax(KAN(Wh_i, Wh_j))
    """
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0., add_self_loops=True,
                 bias=True, use_kan=True, kan_type='fourier', **kwargs):
        super(KAGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.use_kan = use_kan
        self.kan_type = kan_type

        # 线性变换层
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        if use_kan:
            # 使用KAN计算注意力权重
            # 输入: 拼接的节点特征 [h_i || h_j]
            # 输出: 注意力分数
            self.attn_kan = KANLinear(2 * out_channels, 1, kan_type=kan_type)
        else:
            # 原始GAT的注意力机制
            self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
            glorot(self.att)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self, x, edge_index, size=None, return_attention_weights=None):
        """
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
        Returns:
            out: 更新后的节点特征
        """
        H, C = self.heads, self.out_channels

        # 线性变换
        x = self.lin(x).view(-1, H, C)

        # 添加自环
        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # 消息传递
        out = self.propagate(edge_index, x=x, size=size)

        # 多头拼接或平均
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_i, x_j, edge_index_i, size_i):
        """
        计算消息和注意力权重

        Args:
            x_i: 目标节点特征 [num_edges, heads, out_channels]
            x_j: 源节点特征 [num_edges, heads, out_channels]
        """
        if self.use_kan:
            # 使用KAN计算注意力
            # 将多头特征展平，拼接源和目标节点特征
            x_i_flat = x_i.view(-1, self.heads * self.out_channels)
            x_j_flat = x_j.view(-1, self.heads * self.out_channels)

            # 对每个头分别计算注意力
            alpha_list = []
            for h in range(self.heads):
                # 提取当前头的特征
                x_i_h = x_i[:, h, :]  # [num_edges, out_channels]
                x_j_h = x_j[:, h, :]  # [num_edges, out_channels]

                # 拼接并通过KAN
                x_cat = torch.cat([x_i_h, x_j_h], dim=-1)  # [num_edges, 2*out_channels]
                alpha_h = self.attn_kan(x_cat)  # [num_edges, 1]
                alpha_list.append(alpha_h)

            # 合并所有头的注意力分数
            alpha = torch.cat(alpha_list, dim=-1)  # [num_edges, heads]
            alpha = alpha.unsqueeze(-1)  # [num_edges, heads, 1]
        else:
            # 原始GAT的注意力计算
            x_cat = torch.cat([x_i, x_j], dim=-1)  # [num_edges, heads, 2*out_channels]
            alpha = (x_cat * self.att).sum(dim=-1)  # [num_edges, heads]
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = alpha.unsqueeze(-1)  # [num_edges, heads, 1]

        # Softmax归一化
        alpha = softmax(alpha.squeeze(-1), edge_index_i, num_nodes=size_i)
        alpha = alpha.unsqueeze(-1)

        # Dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 加权消息
        return x_j * alpha


class KAGINConv(MessagePassing):
    """
    KAN-enhanced Graph Isomorphism Network (KAGIN)

    核心思想：
    1. 原始GIN使用MLP进行节点特征更新: h' = MLP((1+ε)h + Σh_j)
    2. KAGIN用KAN替代MLP，学习更复杂的聚合函数

    理论基础：
    - GIN的表达能力等价于WL测试
    - KAN的万能逼近能力可以学习任意连续函数
    - 结合两者可以在保持GIN理论性质的同时增强实际表达能力
    """
    def __init__(self, emb_dim, use_kan=True, kan_type='fourier'):
        super(KAGINConv, self).__init__(aggr="add")

        self.emb_dim = emb_dim
        self.use_kan = use_kan
        self.kan_type = kan_type

        if use_kan:
            # 修复5: 添加Pre-normalization
            self.pre_norm = nn.LayerNorm(emb_dim)

            # 使用KAN替代MLP（简化为单层以减少内存使用）
            self.kan = KANLinear(emb_dim, emb_dim, kan_type=kan_type)
            self.batch_norm = nn.BatchNorm1d(emb_dim)
        else:
            # 原始GIN的MLP
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.BatchNorm1d(2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
            )

        self.eps = nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index):
        """
        Args:
            x: 节点特征 [num_nodes, emb_dim]
            edge_index: 边索引 [2, num_edges]
        """
        if self.use_kan:
            # 修复5: Pre-normalization - 在聚合之前归一化输入
            x_norm = self.pre_norm(x)

            # KAN版本（单层）
            # (1 + ε) * x + 聚合邻居消息
            out = (1 + self.eps) * x_norm + self.propagate(edge_index, x=x_norm)

            # 通过单层KAN（内部有LayerNorm）
            out = self.kan(out)
            out = self.batch_norm(out)
        else:
            # 原始GIN
            out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x))

        return out

    def message(self, x_j):
        """简单传递邻居特征"""
        return x_j


class KAGINConvWEdge(MessagePassing):
    """
    带边特征的KAN-enhanced GIN

    在分子图中，边特征（化学键类型）非常重要
    这个版本在消息传递时考虑边特征
    """
    def __init__(self, emb_dim, use_kan=True, kan_type='fourier'):
        super(KAGINConvWEdge, self).__init__(aggr="add")

        self.emb_dim = emb_dim
        self.use_kan = use_kan
        self.kan_type = kan_type

        if use_kan:
            # 使用KAN（简化为单层以减少内存使用）
            self.kan = KANLinear(emb_dim, emb_dim, kan_type=kan_type)
            self.batch_norm = nn.BatchNorm1d(emb_dim)
        else:
            # 原始MLP
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim),
                nn.BatchNorm1d(2 * emb_dim),
                nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim)
            )

        self.eps = nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)

        if self.use_kan:
            out = (1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding)
            out = self.kan(out)
            out = self.batch_norm(out)
        else:
            out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        """
        消息函数：结合邻居特征和边特征

        原始GIN: m_ij = ReLU(h_j + e_ij)
        KAGIN: 可以学习更复杂的组合方式
        """
        return F.relu(x_j + edge_attr)


class KAGCNConv(MessagePassing):
    """
    KAN-enhanced Graph Convolutional Network (KAGCN)

    新的核心思想：
    1. 聚合邻居信息（平均）
    2. 与自身特征平均：(h + mean(邻居)) / 2
    3. 通过KAN进行非线性变换
    4. 与自身特征相加得到下一层：h' = h + KAN((h + mean(邻居)) / 2)

    公式：h' = h + KAN((h + mean(h_j for j in N(i))) / 2)

    优势：
    - 结合自身和邻居信息后再变换
    - KAN学习复杂的聚合模式
    - 残差连接保持梯度流动
    """
    def __init__(self, in_channels, out_channels, use_kan=True, kan_type='fourier'):
        super(KAGCNConv, self).__init__(aggr='mean')  # 使用mean聚合邻居

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_kan = use_kan
        self.kan_type = kan_type

        if use_kan:
            # KAN处理聚合后的信息
            self.kan = KANLinear(in_channels, out_channels, kan_type=kan_type)
        else:
            # baseline使用普通线性层
            self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]

        Returns:
            out: 更新后的节点特征 [num_nodes, out_channels]
        """
        # 1. 聚合邻居信息（平均）
        neighbor_mean = self.propagate(edge_index, x=x)

        # 2. 与自身平均：(h + mean(邻居)) / 2
        combined = (x + neighbor_mean) / 2.0

        # 3. 通过KAN或线性层变换
        if self.use_kan:
            transformed = self.kan(combined)
        else:
            transformed = self.linear(combined)

        # 4. 与自身相加（残差连接）
        # 如果维度不同，需要先投影x
        if self.in_channels != self.out_channels:
            if not hasattr(self, 'skip_connection'):
                self.skip_connection = nn.Linear(self.in_channels, self.out_channels).to(x.device)
            x_skip = self.skip_connection(x)
        else:
            x_skip = x

        out = x_skip + transformed

        return out

    def message(self, x_j):
        """直接传递邻居特征，聚合方式由aggr='mean'控制"""
        return x_j
