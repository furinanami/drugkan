"""
KAN-enhanced GNN Node Layers
完整的KAGNN节点嵌入生成模块，支持多种GNN类型的KAN升级版本
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv, GCNConv, GINConv
from ogb.graphproppred.mol_encoder import AtomEncoder

from .kagnn_conv import KAGATConv, KAGINConv, KAGINConvWEdge, KAGCNConv, KAGCNNeighborConv, KAOriginalConv


class KAGNN_node(nn.Module):
    """
    KAN-enhanced GNN节点嵌入生成器

    设计理念：
    1. 保持与原始GNN_node相同的接口和整体架构
    2. 通过use_kan开关控制是否使用KAN增强
    3. 通过kan_type选择KAN的实现方式（fourier或bspline）

    核心改进：
    - 在每层GNN卷积中使用KAN替代传统的线性变换+激活函数
    - 保留BatchNorm和Dropout等正则化技术
    - 保留残差连接和JK（Jumping Knowledge）机制

    Args:
        num_layer: GNN层数
        emb_dim: 嵌入维度
        drop_ratio: Dropout比率
        JK: Jumping Knowledge模式 ("last", "sum", "multilayer")
        residual: 是否使用残差连接
        gnn_type: GNN类型 ("gat", "gatv2", "gcn", "gin")
        with_edge_attr: 是否使用边特征
        use_kan: 是否使用KAN增强（开关）
        kan_type: KAN类型 ("fourier" 或 "bspline")
    """
    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last",
                 residual=False, gnn_type='gat', with_edge_attr=False,
                 use_kan=False, kan_type='fourier'):
        super(KAGNN_node, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.gnn_type = gnn_type
        self.with_edge_attr = with_edge_attr
        self.use_kan = use_kan
        self.kan_type = kan_type

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # 原子特征编码器（与原始GNN相同）
        self.atom_encoder = AtomEncoder(emb_dim)

        # GNN卷积层列表
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for layer in range(num_layer):
            if use_kan:
                # 使用KAN增强的GNN层
                if with_edge_attr:
                    if gnn_type == 'gin':
                        self.convs.append(KAGINConvWEdge(emb_dim, use_kan=True, kan_type=kan_type))
                    elif gnn_type == 'gcn':
                        # GCN with edge attributes需要特殊处理
                        self.convs.append(KAGCNConv(emb_dim, emb_dim, use_kan=True, kan_type=kan_type))
                    else:
                        raise ValueError(f'KAN-enhanced {gnn_type} with edge attributes not implemented yet')
                else:
                    if gnn_type == 'gin':
                        self.convs.append(KAGINConv(emb_dim, use_kan=True, kan_type=kan_type))
                    elif gnn_type == 'gcn':
                        self.convs.append(KAGCNConv(emb_dim, emb_dim, use_kan=True, kan_type=kan_type))
                    elif gnn_type in ['kagcn_neighbor', 'gcn_neighbor', 'kagcn_msg']:
                        self.convs.append(KAGCNNeighborConv(emb_dim, emb_dim, use_kan=True, kan_type=kan_type))
                    elif gnn_type in ['ka_gnn_original', 'kagnn_original']:
                        self.convs.append(KAOriginalConv(emb_dim, emb_dim, use_kan=True, kan_type=kan_type))
                    elif gnn_type in ['gat', 'gatv2']:
                        self.convs.append(KAGATConv(emb_dim, emb_dim, heads=1, concat=False,
                                                    use_kan=True, kan_type=kan_type))
                    else:
                        raise ValueError(f'Undefined KAN-enhanced GNN type: {gnn_type}')
            else:
                # 使用原始GNN层
                if with_edge_attr:
                    if gnn_type == 'gin':
                        from .conv import GINConvWEdge
                        self.convs.append(GINConvWEdge(emb_dim))
                    elif gnn_type == 'gcn':
                        from .conv import GCNConvWEdge
                        self.convs.append(GCNConvWEdge(emb_dim))
                    else:
                        raise ValueError(f'Undefined GNN type with edge attributes: {gnn_type}')
                else:
                    if gnn_type == 'gin':
                        # GINConv需要一个nn.Module作为参数
                        gin_nn = nn.Sequential(
                            nn.Linear(emb_dim, 2 * emb_dim),
                            nn.BatchNorm1d(2 * emb_dim),
                            nn.ReLU(),
                            nn.Linear(2 * emb_dim, emb_dim)
                        )
                        self.convs.append(GINConv(gin_nn))
                    elif gnn_type == 'gcn':
                        self.convs.append(GCNConv(emb_dim, emb_dim))
                    elif gnn_type in ['kagcn_neighbor', 'gcn_neighbor', 'kagcn_msg']:
                        self.convs.append(KAGCNNeighborConv(emb_dim, emb_dim, use_kan=False, kan_type=kan_type))
                    elif gnn_type in ['ka_gnn_original', 'kagnn_original']:
                        self.convs.append(KAOriginalConv(emb_dim, emb_dim, use_kan=False, kan_type=kan_type))
                    elif gnn_type in ['gat', 'gatv2']:
                        self.convs.append(GATConv(emb_dim, emb_dim))
                    else:
                        raise ValueError(f'Undefined GNN type: {gnn_type}')

            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr):
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, num_features]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边特征 [num_edges, num_edge_features]

        Returns:
            node_representation: 节点表示
                - 如果JK="last": [num_nodes, emb_dim]
                - 如果JK="sum": [num_nodes, emb_dim]
                - 如果JK="multilayer": list of [num_nodes, emb_dim]
        """
        # 初始节点嵌入
        h_list = [self.atom_encoder(x)]

        # 逐层传播
        for layer in range(self.num_layer):
            # GNN卷积
            if edge_attr is None or not self.with_edge_attr:
                h = self.convs[layer](h_list[layer], edge_index)
            else:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            # BatchNorm
            h = self.batch_norms[layer](h)

            # 激活函数和Dropout
            if (layer == self.num_layer - 1) and (not self.JK == "multilayer"):
                # 最后一层不使用激活函数
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                # 使用ELU激活函数
                h = F.dropout(F.elu(h), self.drop_ratio, training=self.training)

            # 残差连接
            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        # Jumping Knowledge聚合
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]
        elif self.JK == "multilayer":
            node_representation = h_list

        return node_representation


class KAGNN_node_Virtualnode(nn.Module):
    """
    带虚拟节点的KAN-enhanced GNN

    虚拟节点机制：
    - 添加一个全局虚拟节点，与所有真实节点相连
    - 虚拟节点聚合全图信息，然后广播回所有节点
    - 增强长程依赖的建模能力

    KAN增强：
    - 在虚拟节点的MLP变换中也可以使用KAN
    - 提升全局信息的表达能力
    """
    def __init__(self, num_layer, emb_dim, drop_ratio=0.5, JK="last",
                 residual=False, gnn_type='gin', use_kan=False, kan_type='fourier'):
        super(KAGNN_node_Virtualnode, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.gnn_type = gnn_type
        self.use_kan = use_kan
        self.kan_type = kan_type

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_encoder = AtomEncoder(emb_dim)

        # 虚拟节点嵌入（初始化为0）
        self.virtualnode_embedding = nn.Embedding(1, emb_dim)
        nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        # GNN卷积层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # 虚拟节点MLP
        self.mlp_virtualnode_list = nn.ModuleList()

        for layer in range(num_layer):
            if use_kan:
                # KAN增强版本
                if gnn_type == 'gin':
                    self.convs.append(KAGINConv(emb_dim, use_kan=True, kan_type=kan_type))
                elif gnn_type == 'gcn':
                    self.convs.append(KAGCNConv(emb_dim, emb_dim, use_kan=True, kan_type=kan_type))
                elif gnn_type in ['kagcn_neighbor', 'gcn_neighbor', 'kagcn_msg']:
                    self.convs.append(KAGCNNeighborConv(emb_dim, emb_dim, use_kan=True, kan_type=kan_type))
                elif gnn_type in ['ka_gnn_original', 'kagnn_original']:
                    self.convs.append(KAOriginalConv(emb_dim, emb_dim, use_kan=True, kan_type=kan_type))
                else:
                    raise ValueError(f'Virtual node not supported for KAN-enhanced {gnn_type}')
            else:
                # 原始版本
                if gnn_type == 'gin':
                    self.convs.append(GINConv(emb_dim))
                elif gnn_type == 'gcn':
                    self.convs.append(GCNConv(emb_dim))
                elif gnn_type in ['kagcn_neighbor', 'gcn_neighbor', 'kagcn_msg']:
                    self.convs.append(KAGCNNeighborConv(emb_dim, emb_dim, use_kan=False, kan_type=kan_type))
                elif gnn_type in ['ka_gnn_original', 'kagnn_original']:
                    self.convs.append(KAOriginalConv(emb_dim, emb_dim, use_kan=False, kan_type=kan_type))
                else:
                    raise ValueError(f'Undefined GNN type: {gnn_type}')

            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

        # 虚拟节点的MLP（可选使用KAN）
        for layer in range(num_layer - 1):
            if use_kan:
                from .kan_layers import KANLinear
                self.mlp_virtualnode_list.append(nn.Sequential(
                    KANLinear(emb_dim, 2 * emb_dim, kan_type=kan_type),
                    nn.BatchNorm1d(2 * emb_dim),
                    KANLinear(2 * emb_dim, emb_dim, kan_type=kan_type),
                    nn.BatchNorm1d(emb_dim)
                ))
            else:
                self.mlp_virtualnode_list.append(nn.Sequential(
                    nn.Linear(emb_dim, 2 * emb_dim),
                    nn.BatchNorm1d(2 * emb_dim),
                    nn.ReLU(),
                    nn.Linear(2 * emb_dim, emb_dim),
                    nn.BatchNorm1d(emb_dim),
                    nn.ReLU()
                ))

    def forward(self, x, edge_index, edge_attr, batch):
        """
        前向传播（带虚拟节点）

        Args:
            x: 节点特征
            edge_index: 边索引
            edge_attr: 边特征
            batch: 批次索引
        """
        from torch_geometric.nn import global_add_pool

        # 初始化虚拟节点嵌入
        virtualnode_embedding = self.virtualnode_embedding(
            torch.zeros(batch[-1].item() + 1).to(edge_index.dtype).to(edge_index.device)
        )

        h_list = [self.atom_encoder(x)]

        for layer in range(self.num_layer):
            # 将虚拟节点信息添加到图节点
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            # 消息传递
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            # 更新虚拟节点
            if layer < self.num_layer - 1:
                virtualnode_embedding_temp = global_add_pool(h_list[layer], batch) + virtualnode_embedding

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio, training=self.training
                    )
                else:
                    virtualnode_embedding = F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.drop_ratio, training=self.training
                    )

        # Jumping Knowledge
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        return node_representation
