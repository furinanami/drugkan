#!/usr/bin/env python3
"""
GraphNN 数据预处理脚本
将CCLE基因表达数据转换为GDSC格式，并完成DrugComb数据预处理
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

print("=" * 70)
print("GraphNN 数据预处理")
print("=" * 70)

# 设置路径
preprocessing_dir = 'data/preprocessing'
raw_data_dir = '/home/bioinfo202200130116/bioinfo/raw_data'

print("\n步骤 1: 加载COSMIC ID映射")
df_cellosaurus = pd.read_csv(f'{preprocessing_dir}/cellosaurus_cosmic_ids.txt',
                              sep=',', header=None).dropna()
dict_cellosaurus = dict(zip(df_cellosaurus[0], df_cellosaurus[1]))
print(f"  ✓ 加载了 {len(dict_cellosaurus)} 个细胞系映射")

print("\n步骤 2: 加载DrugComb药物信息")
df_drugcomb_drugs = pd.read_json(f'{preprocessing_dir}/drugs.json')
dict_smiles = dict(zip(df_drugcomb_drugs.dname, df_drugcomb_drugs.smiles))
print(f"  ✓ 加载了 {len(dict_smiles)} 个药物的SMILES")

print("\n步骤 3: 加载DrugComb协同作用数据")
print("  (这可能需要几分钟...)")
df_drugcomb = pd.read_csv(f'{preprocessing_dir}/summary_v_1_5.csv')
print(f"  ✓ 加载了 {len(df_drugcomb)} 条药物组合记录")

print("\n步骤 4: 添加COSMIC ID")
df_drugcomb["cosmicId"] = [dict_cellosaurus[cell] if cell in dict_cellosaurus.keys()
                           else float('nan') for cell in df_drugcomb['cell_line_name']]

print("\n步骤 5: 清理和过滤数据")
df_drugcomb = df_drugcomb.replace({'\\N':float('nan')}).astype({"synergy_loewe": float})
df_drugcomb = df_drugcomb.dropna(subset=[
    'drug_row', 'drug_col', 'cell_line_name',
    'synergy_zip', 'synergy_loewe', 'synergy_hsa', 'synergy_bliss', 'cosmicId'
])
df_drugcomb = df_drugcomb.astype({"cosmicId": int})
print(f"  ✓ 清理后剩余 {len(df_drugcomb)} 条记录")

print("\n步骤 6: 定义协同作用阈值")
def synergy_threshold(val):
    res = 0
    if (val >= 10.0):
        res = 1
    if (val <= -10.0):
        res = -1
    return res

print("\n步骤 7: 添加SMILES结构")
df_drugcomb["drug_row_smiles"] = [dict_smiles.get(drug, "NULL")
                                   for drug in df_drugcomb.drug_row]
df_drugcomb["drug_col_smiles"] = [dict_smiles.get(drug, "NULL")
                                   for drug in df_drugcomb.drug_col]

# 移除NULL SMILES
null_smiles = df_drugcomb[(df_drugcomb.drug_row_smiles == "NULL") |
                          (df_drugcomb.drug_col_smiles == "NULL")].index
df_drugcomb = df_drugcomb.drop(index=null_smiles)
print(f"  ✓ 移除NULL SMILES后剩余 {len(df_drugcomb)} 条记录")

print("\n步骤 8: 计算协同作用阈值")
df_drugcomb["loewe_thresh"] = [synergy_threshold(val)
                                for val in df_drugcomb.synergy_loewe]
df_drugcomb["zip_thresh"] = [synergy_threshold(val)
                              for val in df_drugcomb.synergy_zip]
df_drugcomb["hsa_thresh"] = [synergy_threshold(val)
                              for val in df_drugcomb.synergy_hsa]
df_drugcomb["bliss_thresh"] = [synergy_threshold(val)
                                for val in df_drugcomb.synergy_bliss]
df_drugcomb["total_thresh"] = df_drugcomb[["loewe_thresh", "zip_thresh",
                                            "hsa_thresh", "bliss_thresh"]].sum(axis=1)

print("\n步骤 9: 选择评分标准并过滤")
score = 'loewe_thresh'
score_val = 1
df_drugcomb_filter = df_drugcomb[df_drugcomb[score].abs() >= score_val].copy()
df_drugcomb_filter['Y'] = [1 if val >= score_val else 0
                           for val in df_drugcomb_filter[score]]
print(f"  ✓ 使用 {score} >= {score_val}, 剩余 {len(df_drugcomb_filter)} 条记录")

print("\n步骤 10: 去除重复记录")
dup_to_drop = []
df_drugcomb_filter_dedup = df_drugcomb_filter.copy()
cols = ['drug_row', 'drug_col', "cell_line_name"]
df_drugcomb_filter_dedup[cols] = np.sort(df_drugcomb_filter_dedup[cols].values, axis=1)
dup = df_drugcomb_filter_dedup.duplicated(subset=cols, keep=False)

dup_score = df_drugcomb_filter_dedup[dup][cols+['Y']]
dup_val = dup_score.duplicated(keep=False)
dup_val_true = df_drugcomb_filter_dedup[dup][cols+['Y']][dup_val]
dup_val_false = df_drugcomb_filter_dedup[dup][cols+['Y']][~dup_val]

dup_to_drop += list(dup_val_true[dup_val_true.duplicated(keep="first")].index)

dup2 = pd.concat([dup_val_false, dup_val_true[~dup_val_true.duplicated(keep="first")]], axis=0)
dup2_val = dup2.duplicated(subset=(cols), keep=False)
dup_to_drop += list(dup2[dup2_val].sort_values(cols).index)

df_drugcomb_filter = df_drugcomb_filter.drop(index=dup_to_drop)
print(f"  ✓ 去重后剩余 {len(df_drugcomb_filter)} 条记录")

print("\n步骤 11: 加载L1000基因信息")
df_l1000 = pd.read_csv(f'{preprocessing_dir}/L1000genes.txt', sep='\t')
# 检查列名
if 'pr_gene_title' in df_l1000.columns:
    # LINCS格式
    df_l1000_lm = df_l1000[df_l1000['pr_is_lm'] == 1]
    lm_genes = list(df_l1000_lm['pr_gene_symbol'])
elif 'Type' in df_l1000.columns:
    df_l1000_lm = df_l1000[df_l1000.Type == "landmark"]
    lm_genes = list(df_l1000_lm.Symbol)
else:
    # 使用所有基因
    lm_genes = list(df_l1000.iloc[:, 0])
print(f"  ✓ 加载了 {len(lm_genes)} 个landmark基因")

print("\n步骤 12: 加载CCLE基因表达数据")
print("  (这可能需要几分钟...)")
df_ccle = pd.read_csv(f'{raw_data_dir}/ccle/OmicsExpressionProteinCodingGenesTPMLogp1.csv',
                      index_col=0)
print(f"  ✓ 加载了 {df_ccle.shape[0]} 个细胞系, {df_ccle.shape[1]} 个基因")

print("\n步骤 13: 转换CCLE数据为GDSC格式")
# 提取基因名称（去除括号中的ID）
gene_names = [col.split(' (')[0] for col in df_ccle.columns]
df_ccle.columns = gene_names

# 转置数据使基因为行
df_ccle_t = df_ccle.T
df_ccle_t['GENE_SYMBOLS'] = df_ccle_t.index

# 只保留landmark基因
df_rma_landm = df_ccle_t[df_ccle_t['GENE_SYMBOLS'].isin(lm_genes)]
print(f"  ✓ 提取了 {len(df_rma_landm)} 个landmark基因的表达数据")

print("\n步骤 14: 保存基因表达数据")
gene_gex = pd.DataFrame(df_rma_landm["GENE_SYMBOLS"].copy())
gene_gex["GEX"] = ["gex" + str(i) for i in range(len(gene_gex))]
gene_gex.to_csv(f'{preprocessing_dir}/gene_gex.tsv', sep='\t', index=False)
print(f"  ✓ 保存基因映射到 gene_gex.tsv")

# 保存landmark基因表达数据
df_rma_landm.to_csv(f'{preprocessing_dir}/df_rma_landm.tsv', sep='\t')
print(f"  ✓ 保存基因表达数据到 df_rma_landm.tsv")

print("\n步骤 15: 重命名列并保存最终数据")
df_drugcomb_filter = df_drugcomb_filter.rename(columns={
    "drug_row": "Drug1_ID",
    "drug_col": "Drug2_ID",
    "cosmicId": "Cosmic_ID",
    "cell_line_name": "Cell_Line_ID",
    "drug_row_smiles": "Drug1",
    "drug_col_smiles": "Drug2"
})

col_sel = [
    'Drug1_ID', 'Drug2_ID', 'Cell_Line_ID', 'Cosmic_ID',
    'Drug1', 'Drug2', 'Y', 'synergy_loewe', 'synergy_zip'
]
output_file = f'{preprocessing_dir}/drugcomb_{score}_{score_val}.csv'
df_drugcomb_filter[col_sel].to_csv(output_file, index=False)
print(f"  ✓ 保存最终数据到 {output_file}")

print("\n步骤 16: 数据统计")
posneg = df_drugcomb_filter.Y.value_counts()
pospercent = round(posneg[1] * 100 / (posneg[1] + posneg[0]), 1)
print(f"  - 总记录数: {len(df_drugcomb_filter)}")
print(f"  - 正样本: {posneg.get(1, 0)} ({pospercent}%)")
print(f"  - 负样本: {posneg.get(0, 0)} ({100-pospercent}%)")
print(f"  - 唯一药物数: {len(set(list(df_drugcomb_filter['Drug1_ID']) + list(df_drugcomb_filter['Drug2_ID'])))}")
print(f"  - 唯一细胞系数: {len(set(df_drugcomb_filter['Cell_Line_ID']))}")

print("\n" + "=" * 70)
print("✅ 数据预处理完成!")
print("=" * 70)
print("\n下一步: 运行 python3 generate_dataset.py 生成训练数据集")
