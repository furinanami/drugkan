#!/usr/bin/env python3
"""
预先生成所有场景的数据分区
生成 random / cold_drug / cold_cell 三种场景的分区文件
"""
import os
import sys
import pickle
import numpy as np
import argparse

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

GRAPHNN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GRAPHNN_DIR)

from deepadr.dataset import MoleculeDataset
from deepadr.cold_split import get_split_by_scenario


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DrugComb split files")
    parser.add_argument(
        "--dataset-root",
        default=os.path.join(GRAPHNN_DIR, "data/processed/DrugComb_loewe_thresh_1/data_v1"),
        help="MoleculeDataset root",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument(
        "--scenarios",
        default="random,cold_drug,cold_cell",
        help="Comma-separated scenarios to generate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("生成数据分区文件")
    print("=" * 80)

    dataset_path = args.dataset_root

    # 加载数据集
    print("\n加载数据集...")
    dataset = MoleculeDataset(root=dataset_path, dataset='tdcSynergy')
    print(f"数据集大小: {len(dataset)}")

    # 创建分区保存目录
    partition_dir = os.path.join(dataset_path, "partitions")
    os.makedirs(partition_dir, exist_ok=True)

    # 生成指定场景的分区
    scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]

    for scenario in scenarios:
        print(f"\n生成 {scenario} 场景分区...")

        partition = get_split_by_scenario(
            dataset,
            scenario=scenario,
            fold=args.fold,
            n_folds=5,
            seed=args.seed
        )

        # 统一键名（valid -> validation）
        if 'valid' in partition and 'validation' not in partition:
            partition['validation'] = partition['valid']
            del partition['valid']

        print(f"  训练集: {len(partition['train'])}")
        print(f"  验证集: {len(partition['validation'])}")
        print(f"  测试集: {len(partition['test'])}")
        if scenario == "cold_drug" and partition.get("split_strategy"):
            print(f"  策略: {partition['split_strategy']}")
            print(f"  Held-out test drugs: {len(partition.get('heldout_test_drugs', []))}")
            print(f"  Held-out validation drugs: {len(partition.get('heldout_validation_drugs', []))}")

        # 保存分区
        partition_file = os.path.join(partition_dir, f"partition_{scenario}.pkl")
        with open(partition_file, 'wb') as f:
            pickle.dump(partition, f)

        print(f"  ✓ 保存到: {partition_file}")

    print("\n" + "=" * 80)
    print("✅ 所有分区文件生成完成！")
    print("=" * 80)
    print(f"\n分区文件保存在: {partition_dir}/")
    for scenario in scenarios:
        print(f"  - partition_{scenario}.pkl")


if __name__ == '__main__':
    main()
