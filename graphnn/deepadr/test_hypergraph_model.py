"""
测试脚本：验证DeepDDS超图模型的实现
"""

import torch
import numpy as np
from torch_geometric.data import Data, Batch

# 导入模型
from model_hypergraph import (
    DeepDDS_Hypergraph,
    DeepDDS_Hypergraph_WithReconstruction,
    EmbeddingAligner,
    HypergraphEncoder,
    HypergraphDecoder,
    BatchHypergraphBuilder
)


def create_dummy_batch(batch_size=8, num_drugs=20, num_cells=10):
    """创建虚拟批次数据用于测试"""

    data_list = []

    for i in range(batch_size):
        # 随机选择药物和细胞系ID
        drug_a_id = np.random.randint(0, num_drugs)
        drug_b_id = np.random.randint(0, num_drugs)
        cell_id = np.random.randint(0, num_cells)

        # 随机标签（50%正样本）
        label = np.random.randint(0, 2)

        # 创建虚拟分子图（药物A）
        num_atoms_a = np.random.randint(10, 30)
        x_a = torch.randn(num_atoms_a, 9)  # 9维原子特征
        edge_index_a = torch.randint(0, num_atoms_a, (2, num_atoms_a * 2))
        edge_attr_a = torch.randn(num_atoms_a * 2, 1)

        # 创建虚拟分子图（药物B）
        num_atoms_b = np.random.randint(10, 30)
        x_b = torch.randn(num_atoms_b, 9)
        edge_index_b = torch.randint(0, num_atoms_b, (2, num_atoms_b * 2))
        edge_attr_b = torch.randn(num_atoms_b * 2, 1)

        # 创建虚拟基因表达
        expression = torch.randn(926)

        # 创建Data对象
        data = Data(
            x_a=x_a,
            edge_index_a=edge_index_a,
            edge_attr_a=edge_attr_a,
            x_b=x_b,
            edge_index_b=edge_index_b,
            edge_attr_b=edge_attr_b,
            expression=expression,
            drug_a_id=torch.tensor([drug_a_id], dtype=torch.long),
            drug_b_id=torch.tensor([drug_b_id], dtype=torch.long),
            cell_id=torch.tensor([cell_id], dtype=torch.long),
            y=torch.tensor([label], dtype=torch.long),
            id=torch.tensor([i], dtype=torch.long)
        )

        data_list.append(data)

    # 创建批次
    batch = Batch.from_data_list(data_list, follow_batch=['x_a', 'x_b'])

    return batch


def test_embedding_aligner():
    """测试嵌入对齐模块"""
    print("\n" + "="*50)
    print("测试 EmbeddingAligner")
    print("="*50)

    aligner = EmbeddingAligner(drug_dim=128, cell_dim=256, output_dim=256)

    drug_emb = torch.randn(16, 128)
    cell_emb = torch.randn(16, 256)

    drug_aligned, cell_aligned = aligner(drug_emb, cell_emb)

    print(f"✓ 输入药物嵌入: {drug_emb.shape}")
    print(f"✓ 输入细胞嵌入: {cell_emb.shape}")
    print(f"✓ 输出药物嵌入: {drug_aligned.shape}")
    print(f"✓ 输出细胞嵌入: {cell_aligned.shape}")

    assert drug_aligned.shape == (16, 256), "药物嵌入维度错误"
    assert cell_aligned.shape == (16, 256), "细胞嵌入维度错误"

    print("✓ EmbeddingAligner 测试通过！")


def test_hypergraph_builder():
    """测试超图构建器"""
    print("\n" + "="*50)
    print("测试 BatchHypergraphBuilder")
    print("="*50)

    batch = create_dummy_batch(batch_size=8)

    drug_emb_a = torch.randn(8, 256)
    drug_emb_b = torch.randn(8, 256)
    cell_emb = torch.randn(8, 256)

    builder = BatchHypergraphBuilder()
    node_features, hyperedge_index, node_mapping, batch_node_indices = \
        builder.build_hypergraph(batch, drug_emb_a, drug_emb_b, cell_emb)

    print(f"✓ 节点特征: {node_features.shape}")
    print(f"✓ 超边索引: {hyperedge_index.shape}")
    print(f"✓ 节点映射数量: {len(node_mapping)}")
    print(f"✓ 正样本数量: {(batch.y == 1).sum().item()}")
    print(f"✓ 超边数量: {hyperedge_index.shape[1] // 3}")

    assert node_features.shape[1] == 256, "节点特征维度错误"
    assert hyperedge_index.shape[0] == 2, "超边索引格式错误"

    print("✓ BatchHypergraphBuilder 测试通过！")


def test_hypergraph_encoder():
    """测试超图编码器"""
    print("\n" + "="*50)
    print("测试 HypergraphEncoder")
    print("="*50)

    encoder = HypergraphEncoder(in_channels=256, hidden_channels=512, out_channels=256)

    # 测试有超边的情况
    node_features = torch.randn(30, 256)
    hyperedge_index = torch.randint(0, 30, (2, 60))

    output = encoder(node_features, hyperedge_index)

    print(f"✓ 输入节点特征: {node_features.shape}")
    print(f"✓ 超边索引: {hyperedge_index.shape}")
    print(f"✓ 输出节点嵌入: {output.shape}")

    assert output.shape == (30, 256), "输出维度错误"

    # 测试无超边的情况（fallback）
    empty_hyperedge_index = torch.zeros(2, 0, dtype=torch.long)
    output_fallback = encoder(node_features, empty_hyperedge_index)

    print(f"✓ Fallback输出: {output_fallback.shape}")
    assert output_fallback.shape == (30, 256), "Fallback输出维度错误"

    print("✓ HypergraphEncoder 测试通过！")


def test_hypergraph_decoder():
    """测试解码器"""
    print("\n" + "="*50)
    print("测试 HypergraphDecoder")
    print("="*50)

    decoder = HypergraphDecoder(in_channels=768, hidden1=512, hidden2=256, num_classes=2)

    h_a = torch.randn(16, 256)
    h_b = torch.randn(16, 256)
    h_e = torch.randn(16, 256)

    log_probs = decoder(h_a, h_b, h_e)

    print(f"✓ 输入维度: {h_a.shape}, {h_b.shape}, {h_e.shape}")
    print(f"✓ 输出log概率: {log_probs.shape}")
    print(f"✓ 概率和: {torch.exp(log_probs).sum(dim=1)[:5]}")

    assert log_probs.shape == (16, 2), "输出维度错误"
    assert torch.allclose(torch.exp(log_probs).sum(dim=1), torch.ones(16), atol=1e-5), "概率和不为1"

    print("✓ HypergraphDecoder 测试通过！")


def test_deepdds_hypergraph_mode():
    """测试超图模式"""
    print("\n" + "="*50)
    print("测试 DeepDDS_Hypergraph (use_hypergraph=True)")
    print("="*50)

    model = DeepDDS_Hypergraph(
        num_features_xd=9,
        gat_output_dim=128,
        expression_input_size=926,
        unified_dim=256,
        hypergraph_hidden=512,
        use_hypergraph=True
    )

    batch = create_dummy_batch(batch_size=8)

    model.eval()
    with torch.no_grad():
        log_probs = model(batch)

    print(f"✓ 批次大小: {len(batch.y)}")
    print(f"✓ 输出形状: {log_probs.shape}")
    print(f"✓ 预测类别: {torch.argmax(log_probs, dim=1)}")
    print(f"✓ 真实标签: {batch.y.squeeze()}")

    assert log_probs.shape == (8, 2), "输出维度错误"

    print("✓ DeepDDS_Hypergraph (超图模式) 测试通过！")


def test_deepdds_mlp_mode():
    """测试简单MLP模式"""
    print("\n" + "="*50)
    print("测试 DeepDDS_Hypergraph (use_hypergraph=False)")
    print("="*50)

    model = DeepDDS_Hypergraph(
        num_features_xd=9,
        gat_output_dim=128,
        expression_input_size=926,
        unified_dim=256,
        use_hypergraph=False  # 关闭超图
    )

    batch = create_dummy_batch(batch_size=8)

    model.eval()
    with torch.no_grad():
        log_probs = model(batch)

    print(f"✓ 批次大小: {len(batch.y)}")
    print(f"✓ 输出形状: {log_probs.shape}")
    print(f"✓ 预测类别: {torch.argmax(log_probs, dim=1)}")

    assert log_probs.shape == (8, 2), "输出维度错误"

    print("✓ DeepDDS_Hypergraph (MLP模式) 测试通过！")


def test_deepdds_with_reconstruction():
    """测试带重构损失的模型"""
    print("\n" + "="*50)
    print("测试 DeepDDS_Hypergraph_WithReconstruction")
    print("="*50)

    model = DeepDDS_Hypergraph_WithReconstruction(
        num_features_xd=9,
        gat_output_dim=128,
        expression_input_size=926,
        unified_dim=256,
        hypergraph_hidden=512,
        use_hypergraph=True
    )

    batch = create_dummy_batch(batch_size=8)

    model.eval()
    with torch.no_grad():
        log_probs, rec_drug, rec_cline = model(batch)

    print(f"✓ 输出形状: {log_probs.shape}")
    print(f"✓ 药物重构矩阵: {rec_drug.shape if rec_drug is not None else 'None'}")
    print(f"✓ 细胞系重构矩阵: {rec_cline.shape if rec_cline is not None else 'None'}")

    assert log_probs.shape == (8, 2), "输出维度错误"
    if rec_drug is not None:
        assert rec_drug.shape[0] == rec_drug.shape[1], "重构矩阵应该是方阵"

    print("✓ DeepDDS_Hypergraph_WithReconstruction 测试通过！")


def test_backward_pass():
    """测试反向传播"""
    print("\n" + "="*50)
    print("测试反向传播和梯度")
    print("="*50)

    model = DeepDDS_Hypergraph(
        num_features_xd=9,
        gat_output_dim=128,
        expression_input_size=926,
        unified_dim=256,
        use_hypergraph=True
    )

    batch = create_dummy_batch(batch_size=8)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.NLLLoss()

    # 前向传播
    log_probs = model(batch)
    loss = criterion(log_probs, batch.y.squeeze())

    print(f"✓ 损失值: {loss.item():.4f}")

    # 反向传播
    optimizer.zero_grad()
    loss.backward()

    # 检查梯度
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "没有计算梯度"
    print("✓ 梯度计算正常")

    # 优化器步骤
    optimizer.step()
    print("✓ 优化器更新正常")

    print("✓ 反向传播测试通过！")


def test_mode_comparison():
    """对比两种模式的输出"""
    print("\n" + "="*50)
    print("对比超图模式 vs MLP模式")
    print("="*50)

    batch = create_dummy_batch(batch_size=8)

    # 超图模式
    model_hg = DeepDDS_Hypergraph(
        num_features_xd=9,
        gat_output_dim=128,
        expression_input_size=926,
        unified_dim=256,
        use_hypergraph=True
    )

    # MLP模式
    model_mlp = DeepDDS_Hypergraph(
        num_features_xd=9,
        gat_output_dim=128,
        expression_input_size=926,
        unified_dim=256,
        use_hypergraph=False
    )

    model_hg.eval()
    model_mlp.eval()

    with torch.no_grad():
        output_hg = model_hg(batch)
        output_mlp = model_mlp(batch)

    print(f"✓ 超图模式输出: {output_hg.shape}")
    print(f"✓ MLP模式输出: {output_mlp.shape}")
    print(f"✓ 超图模式预测: {torch.argmax(output_hg, dim=1)}")
    print(f"✓ MLP模式预测: {torch.argmax(output_mlp, dim=1)}")

    # 统计参数量
    params_hg = sum(p.numel() for p in model_hg.parameters())
    params_mlp = sum(p.numel() for p in model_mlp.parameters())

    print(f"✓ 超图模式参数量: {params_hg:,}")
    print(f"✓ MLP模式参数量: {params_mlp:,}")
    print(f"✓ 参数差异: {params_hg - params_mlp:,} ({(params_hg/params_mlp - 1)*100:.1f}%)")

    print("✓ 模式对比测试通过！")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print(" "*15 + "DeepDDS超图模型测试套件")
    print("="*70)

    try:
        test_embedding_aligner()
        test_hypergraph_builder()
        test_hypergraph_encoder()
        test_hypergraph_decoder()
        test_deepdds_hypergraph_mode()
        test_deepdds_mlp_mode()
        test_deepdds_with_reconstruction()
        test_backward_pass()
        test_mode_comparison()

        print("\n" + "="*70)
        print(" "*20 + "✓ 所有测试通过！")
        print("="*70)

    except Exception as e:
        print("\n" + "="*70)
        print(" "*20 + "✗ 测试失败！")
        print("="*70)
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    run_all_tests()
