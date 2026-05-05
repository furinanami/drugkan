#!/usr/bin/env python3
"""
预先生成所有场景的数据分区
生成 random / cold_drug / cold_cell 三种场景的分区文件
"""
import os
import sys
import pickle
import numpy as np

sys.path.insert(0, '/home/bioinfo202200130116/bioinfo/codebasev5/graphnn')

from deepadr.dataset import MoleculeDataset
from deepadr.cold_split import get_split_by_scenario


def main():
    print("=" * 80)
    print("生成数据分区文件")
    print("=" * 80)

    # 数据集路径
    dataset_path = "data/processed/DrugComb_loewe_thresh_1/data_v1"

    # 加载数据集
    print("\n加载数据集...")
    dataset = MoleculeDataset(root=dataset_path, dataset='tdcSynergy')
    print(f"数据集大小: {len(dataset)}")

    # 创建分区保存目录
    partition_dir = os.path.join(dataset_path, "partitions")
    os.makedirs(partition_dir, exist_ok=True)

    # 生成三种场景的分区
    scenarios = ['random', 'cold_drug', 'cold_cell']

    for scenario in scenarios:
        print(f"\n生成 {scenario} 场景分区...")

        partition = get_split_by_scenario(
            dataset,
            scenario=scenario,
            fold=0,
            n_folds=5,
            seed=42
        )

        # 统一键名（valid -> validation）
        if 'valid' in partition and 'validation' not in partition:
            partition['validation'] = partition['valid']
            del partition['valid']

        print(f"  训练集: {len(partition['train'])}")
        print(f"  验证集: {len(partition['validation'])}")
        print(f"  测试集: {len(partition['test'])}")

        # 保存分区
        partition_file = os.path.join(partition_dir, f"partition_{scenario}.pkl")
        with open(partition_file, 'wb') as f:
            pickle.dump(partition, f)

        print(f"  ✓ 保存到: {partition_file}")

    print("\n" + "=" * 80)
    print("✅ 所有分区文件生成完成！")
    print("=" * 80)
    print(f"\n分区文件保存在: {partition_dir}/")
    print("  - partition_random.pkl")
    print("  - partition_cold_drug.pkl")
    print("  - partition_cold_cell.pkl")


if __name__ == '__main__':
    main()
