#!/usr/bin/env python3
"""
GraphNN 数据集生成脚本（优化版）
按细胞系和药物组织数据，批量生成训练集
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

# 添加deepadr到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deepadr.dataset import *
from deepadr.utilities import *
from deepadr.chemfeatures import *

print("=" * 70)
print("GraphNN 数据集生成（优化版）")
print("=" * 70)

# 设置路径
preprocessing_dir = 'data/preprocessing'
processed_dir = 'data/processed'
os.makedirs(processed_dir, exist_ok=True)

print("\n步骤 1: 加载预处理的DrugComb数据")
score = 'loewe_thresh'
score_val = 1
drugcomb_file = f'{preprocessing_dir}/drugcomb_{score}_{score_val}.csv'
df_data = pd.read_csv(drugcomb_file)
print(f"  ✓ 加载了 {len(df_data)} 条药物组合记录")

print("\n步骤 2: 提取唯一药物并转换为分子图")
drugs_1 = df_data[['Drug1_ID', 'Drug1']].rename(columns={'Drug1_ID': 'Drug_ID', 'Drug1': 'SMILES'})
drugs_2 = df_data[['Drug2_ID', 'Drug2']].rename(columns={'Drug2_ID': 'Drug_ID', 'Drug2': 'SMILES'})
uniq_drugs = pd.concat([drugs_1, drugs_2]).drop_duplicates(subset=['Drug_ID']).reset_index(drop=True)
print(f"  ✓ 找到 {len(uniq_drugs)} 个唯一药物")

print("\n  转换SMILES为分子图...")
mol_graphs = {}
failed_drugs = []

for idx, row in tqdm(uniq_drugs.iterrows(), total=len(uniq_drugs), desc="  处理药物"):
    drug_id = row['Drug_ID']
    smiles = row['SMILES']

    try:
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            failed_drugs.append(drug_id)
            continue

        # 提取原子特征（使用OGB标准特征）
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(atom_to_feature_vector(atom))

        x = torch.tensor(atom_features, dtype=torch.long)

        # 提取边
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])

            # 使用OGB标准边特征
            bond_feature = bond_to_feature_vector(bond)
            edge_attr.append(bond_feature)
            edge_attr.append(bond_feature)

        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 3), dtype=torch.long)  # OGB边特征是3维的

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        mol_graphs[drug_id] = data

    except Exception as e:
        failed_drugs.append(drug_id)
        continue

print(f"  ✓ 成功转换 {len(mol_graphs)} 个药物")
if failed_drugs:
    print(f"  ⚠ 失败 {len(failed_drugs)} 个药物")

print("\n步骤 3: 过滤数据并按细胞系组织")
df_data_filtered = df_data[
    df_data['Drug1_ID'].isin(mol_graphs.keys()) &
    df_data['Drug2_ID'].isin(mol_graphs.keys())
].reset_index(drop=True)
print(f"  ✓ 过滤后剩余 {len(df_data_filtered)} 条记录")

# 按细胞系分组
grouped_by_cell = df_data_filtered.groupby('Cell_Line_ID')
print(f"  ✓ 数据分布在 {len(grouped_by_cell)} 个细胞系")

print("\n步骤 4: 加载基因表达数据")
df_rma = pd.read_csv(f'{preprocessing_dir}/df_rma_landm.tsv', sep='\t', index_col=0)
if 'GENE_SYMBOLS' in df_rma.columns:
    df_rma = df_rma.drop(columns=['GENE_SYMBOLS'])
print(f"  ✓ 加载了 {df_rma.shape[0]} 个基因, {df_rma.shape[1]} 个细胞系")

# 计算默认表达（所有细胞系的平均值）
numeric_cols = df_rma.select_dtypes(include=[np.number]).columns
default_expr = df_rma[numeric_cols].mean(axis=1).values.astype(np.float32)
print(f"  ✓ 计算了默认基因表达向量")

print("\n步骤 5: 按细胞系批量创建数据集")
X_pairs = {}
y_labels = {}
expressions = {}
drug_ids = {}  # 新增：保存每个样本的药物ID
cell_ids = {}  # 新增：保存每个样本的细胞系ID
loewe_scores = {}  # 连续Loewe协同分数
zip_scores = {}  # 连续ZIP协同分数

sample_idx = 0
for cell_line, group in tqdm(grouped_by_cell, desc="  处理细胞系"):
    # 获取该细胞系的基因表达
    matched_col = None
    for col in df_rma.columns:
        if cell_line in col or col in cell_line:
            matched_col = col
            break

    if matched_col:
        cell_expr = df_rma[matched_col].values.astype(np.float32)
    else:
        cell_expr = default_expr

    # 批量处理该细胞系的所有药物对
    for _, row in group.iterrows():
        drug1_id = row['Drug1_ID']
        drug2_id = row['Drug2_ID']
        label = row['Y']

        # 创建药物对
        data_a = mol_graphs[drug1_id]
        data_b = mol_graphs[drug2_id]
        pair_data = PairData(data_a, data_b)

        X_pairs[sample_idx] = pair_data
        y_labels[sample_idx] = label
        expressions[sample_idx] = cell_expr
        drug_ids[sample_idx] = (drug1_id, drug2_id)  # 保存药物ID对
        cell_ids[sample_idx] = cell_line  # 保存细胞系ID
        loewe_scores[sample_idx] = float(row['synergy_loewe'])
        zip_scores[sample_idx] = float(row['synergy_zip'])

        sample_idx += 1

print(f"  ✓ 创建了 {len(X_pairs)} 个训练样本")

print("\n步骤 6: 保存数据集")
# 保存到processed目录（用于直接访问）
with open(f'{processed_dir}/X_pairs.pkl', 'wb') as f:
    pickle.dump(X_pairs, f)
print(f"  ✓ 保存药物对数据")

with open(f'{processed_dir}/y_labels.pkl', 'wb') as f:
    pickle.dump(y_labels, f)
print(f"  ✓ 保存标签")

with open(f'{processed_dir}/expressions.pkl', 'wb') as f:
    pickle.dump(expressions, f)
print(f"  ✓ 保存基因表达")

with open(f'{processed_dir}/drug_ids.pkl', 'wb') as f:
    pickle.dump(drug_ids, f)
print(f"  ✓ 保存药物ID")

with open(f'{processed_dir}/cell_ids.pkl', 'wb') as f:
    pickle.dump(cell_ids, f)
print(f"  ✓ 保存细胞系ID")

with open(f'{processed_dir}/loewe_scores.pkl', 'wb') as f:
    pickle.dump(loewe_scores, f)
print(f"  ✓ 保存Loewe协同分数")

with open(f'{processed_dir}/zip_scores.pkl', 'wb') as f:
    pickle.dump(zip_scores, f)
print(f"  ✓ 保存ZIP协同分数")

# 同时保存到PyG数据集格式的raw目录（用于MoleculeDataset加载）
pyg_dataset_dir = f'{processed_dir}/DrugComb_loewe_thresh_1/data_v1'
os.makedirs(f'{pyg_dataset_dir}/raw', exist_ok=True)

with open(f'{pyg_dataset_dir}/raw/X.pkl', 'wb') as f:
    pickle.dump(X_pairs, f)

with open(f'{pyg_dataset_dir}/raw/y.pkl', 'wb') as f:
    pickle.dump(y_labels, f)

with open(f'{pyg_dataset_dir}/raw/expression.pkl', 'wb') as f:
    pickle.dump(expressions, f)

with open(f'{pyg_dataset_dir}/raw/drug_ids.pkl', 'wb') as f:
    pickle.dump(drug_ids, f)

with open(f'{pyg_dataset_dir}/raw/cell_ids.pkl', 'wb') as f:
    pickle.dump(cell_ids, f)

with open(f'{pyg_dataset_dir}/raw/loewe_scores.pkl', 'wb') as f:
    pickle.dump(loewe_scores, f)

with open(f'{pyg_dataset_dir}/raw/zip_scores.pkl', 'wb') as f:
    pickle.dump(zip_scores, f)

print(f"  ✓ 保存PyG数据集格式文件到 {pyg_dataset_dir}/raw/")

print("\n步骤 7: 创建数据分区（5折交叉验证）")
y_array = np.array([y_labels[i] for i in range(len(y_labels))])
data_partitions = get_stratified_partitions(y_array, num_folds=5,
                                           valid_set_portion=0.1,
                                           random_state=42)

with open(f'{processed_dir}/data_partitions.pkl', 'wb') as f:
    pickle.dump(data_partitions, f)
print(f"  ✓ 保存数据分区")

print("\n步骤 8: 数据集统计")
print(f"  - 总样本数: {len(X_pairs)}")
print(f"  - 正样本: {sum(y_array)} ({sum(y_array)/len(y_array)*100:.1f}%)")
print(f"  - 负样本: {len(y_array)-sum(y_array)} ({(len(y_array)-sum(y_array))/len(y_array)*100:.1f}%)")
print(f"  - 特征维度: {list(expressions.values())[0].shape[0]}")
print(f"  - 唯一细胞系: {len(grouped_by_cell)}")
print(f"  - 唯一药物: {len(mol_graphs)}")

print("\n" + "=" * 70)
print("✅ 数据集生成完成!")
print("=" * 70)
print("\n下一步: 运行 python3 train_model.py 训练模型")
