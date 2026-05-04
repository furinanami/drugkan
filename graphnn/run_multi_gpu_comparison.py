#!/usr/bin/env python3
"""
多GPU并行对比实验
同时运行多种GNN类型的baseline和KAGNN版本对比

使用方法：
    python3 run_multi_gpu_comparison.py --gpus 1,2,3,4,5,6 --epochs 100
"""
import os
import sys
import torch
import torch.multiprocessing as mp
import argparse
import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, '/home/bioinfo202200130116/bioinfo/codebasev5/graphnn')

from deepadr.dataset import MoleculeDataset, get_stratified_partitions
from deepadr.utilities import create_directory
from deepadr.train_functions_kagnn import run_exp_kagnn


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-GPU GNN vs KAGNN vs Hypergraph Comparison')

    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                       help='Comma-separated GPU IDs (default: 0,1,2,3)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--emb_dim', type=int, default=128,
                       help='Embedding dimension (default: 128)')
    parser.add_argument('--num_layer', type=int, default=5,
                       help='Number of GNN layers (default: 5)')
    parser.add_argument('--gnn_types', type=str, default='gin,gcn',
                       help='Comma-separated GNN types to train (gin,gcn,gatv2). Default: gin,gcn')
    parser.add_argument('--use_kan', type=str, default='true',
                       choices=['true', 'false', 'both'],
                       help='Whether to use KAN: true (only KAN), false (only baseline), both (train both). Default: true')
    parser.add_argument('--use_hypergraph', type=str, default='both',
                       choices=['true', 'false', 'both'],
                       help='Whether to use hypergraph: true (only hypergraph), false (only baseline), both (train both). Default: both')
    parser.add_argument('--kan_type', type=str, default='fourier',
                       choices=['fourier', 'bspline'],
                       help='KAN type (default: fourier)')

    return parser.parse_args()


def create_training_params(args, gnn_type, use_kan, use_hypergraph):
    """创建训练参数"""
    tp = {
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,

        "emb_dim": args.emb_dim,
        "gnn_type": gnn_type,
        "num_layer": args.num_layer,
        "graph_pooling": "mean",

        # KAN参数
        "use_kan": use_kan,
        "kan_type": args.kan_type if use_kan else None,

        # 超图参数
        "use_hypergraph": use_hypergraph,
        "use_reconstruction": False,
        "unified_dim": 128,
        "hypergraph_hidden": 256,
        "decoder_hidden1": 256,
        "decoder_hidden2": 128,

        "num_attn_heads": 10,
        "p_dropout": 0.2,

        "base_lr": 1e-4,
        "max_lr_mul": 3,
        "l2_reg": 1e-4,

        "expression_input_size": 946,
        "exp_H1": 8192,
        "exp_H2": 4096
    }
    return tp


def run_single_experiment_wrapper(args_tuple):
    """
    包装函数用于多进程
    """
    (dataset_path, partition, tp, exp_name, exp_dir, gpu_id) = args_tuple

    try:
        # 重新加载数据集（每个进程需要自己的数据集实例）
        dataset = MoleculeDataset(root=dataset_path, dataset='tdcSynergy')

        # 根据是否使用超图选择训练函数
        queue = mp.Queue()

        if tp.get('use_hypergraph', False):
            from deepadr.train_functions_hypergraph import run_exp_deepdds_hypergraph
            run_exp_deepdds_hypergraph(queue, dataset, gpu_id, tp, exp_dir, partition)
        else:
            from deepadr.train_functions_kagnn import run_exp_kagnn
            run_exp_kagnn(queue, dataset, gpu_id, tp, exp_dir, partition)

        # 读取结果
        curves_df = pd.read_csv(os.path.join(exp_dir, "curves.csv"), index_col=0)

        # 获取最佳性能
        best_epoch = curves_df['test_aupr'].idxmax()
        best_results = {
            'exp_name': exp_name,
            'exp_dir': exp_dir,
            'gpu_id': gpu_id,
            'best_epoch': best_epoch,
            'test_aupr': curves_df.loc[best_epoch, 'test_aupr'],
            'test_auc': curves_df.loc[best_epoch, 'test_auc'],
            'valid_aupr': curves_df.loc[best_epoch, 'valid_aupr'],
            'valid_auc': curves_df.loc[best_epoch, 'valid_auc'],
            'use_kan': tp.get('use_kan', False),
            'use_hypergraph': tp.get('use_hypergraph', False),
        }

        print(f"\n[GPU {gpu_id}] 实验 {exp_name} 完成!")
        print(f"[GPU {gpu_id}] 最佳epoch: {best_epoch}")
        print(f"[GPU {gpu_id}] Test AUPR: {best_results['test_aupr']:.4f}")
        print(f"[GPU {gpu_id}] Test AUC: {best_results['test_auc']:.4f}")

        return best_results, curves_df

    except Exception as e:
        print(f"\n[GPU {gpu_id}] 实验 {exp_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def plot_comparison_by_gnn_type(results_by_type, output_dir):
    """为每种GNN类型绘制对比图"""

    for gnn_type, results in results_by_type.items():
        if len(results) < 2:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # AUPR对比
        ax = axes[0]
        exp_names = [r['exp_name'] for r in results]
        aupr_values = [r['test_aupr'] for r in results]

        bars = ax.bar(range(len(exp_names)), aupr_values, alpha=0.8)
        best_idx = aupr_values.index(max(aupr_values))
        bars[best_idx].set_color('green')

        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.set_ylabel('Test AUPR')
        ax.set_title(f'{gnn_type.upper()} - Test AUPR Comparison')
        ax.grid(axis='y', alpha=0.3)

        for i, v in enumerate(aupr_values):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')

        # AUC对比
        ax = axes[1]
        auc_values = [r['test_auc'] for r in results]

        bars = ax.bar(range(len(exp_names)), auc_values, alpha=0.8)
        best_idx = auc_values.index(max(auc_values))
        bars[best_idx].set_color('green')

        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.set_ylabel('Test AUC')
        ax.set_title(f'{gnn_type.upper()} - Test AUC Comparison')
        ax.grid(axis='y', alpha=0.3)

        for i, v in enumerate(auc_values):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{gnn_type}.png'), dpi=150)
        plt.close()


def plot_overall_comparison(all_results, output_dir):
    """绘制总体对比图"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 按GNN类型分组
    gnn_types = sorted(set([r['exp_name'].split('_')[0] for r in all_results]))

    # Test AUPR对比
    ax = axes[0, 0]
    x_pos = 0
    x_labels = []
    x_ticks = []

    for gnn_type in gnn_types:
        type_results = [r for r in all_results if r['exp_name'].startswith(gnn_type)]

        for i, result in enumerate(type_results):
            color = 'lightblue' if 'baseline' in result['exp_name'].lower() else 'orange'
            ax.bar(x_pos, result['test_aupr'], color=color, alpha=0.8)
            x_labels.append(result['exp_name'])
            x_ticks.append(x_pos)
            x_pos += 1

        x_pos += 0.5  # 组间间隔

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Test AUPR')
    ax.set_title('Test AUPR - All Models')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(['Baseline', 'KAGNN'], loc='upper right')

    # Test AUC对比
    ax = axes[0, 1]
    x_pos = 0

    for gnn_type in gnn_types:
        type_results = [r for r in all_results if r['exp_name'].startswith(gnn_type)]

        for i, result in enumerate(type_results):
            color = 'lightblue' if 'baseline' in result['exp_name'].lower() else 'orange'
            ax.bar(x_pos, result['test_auc'], color=color, alpha=0.8)
            x_pos += 1

        x_pos += 0.5

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Test AUC')
    ax.set_title('Test AUC - All Models')
    ax.grid(axis='y', alpha=0.3)

    # 性能提升百分比
    ax = axes[1, 0]
    improvements = []
    labels = []

    for gnn_type in gnn_types:
        type_results = [r for r in all_results if r['exp_name'].startswith(gnn_type)]
        baseline = next((r for r in type_results if 'baseline' in r['exp_name'].lower()), None)
        kagnn = next((r for r in type_results if 'kagnn' in r['exp_name'].lower()), None)

        if baseline and kagnn:
            aupr_improve = (kagnn['test_aupr'] - baseline['test_aupr']) / baseline['test_aupr'] * 100
            improvements.append(aupr_improve)
            labels.append(gnn_type.upper())

    colors = ['green' if x > 0 else 'red' for x in improvements]
    ax.bar(range(len(improvements)), improvements, color=colors, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('AUPR Improvement (%)')
    ax.set_title('KAGNN vs Baseline - AUPR Improvement')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    for i, v in enumerate(improvements):
        ax.text(i, v, f'{v:+.2f}%', ha='center', va='bottom' if v > 0 else 'top')

    # 训练时间对比（如果有的话）
    ax = axes[1, 1]
    ax.text(0.5, 0.5, 'Training Time Comparison\n(To be implemented)',
            ha='center', va='center', fontsize=12)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_comparison.png'), dpi=150)
    plt.close()


def generate_report(all_results, output_dir):
    """生成详细报告"""

    report_path = os.path.join(output_dir, 'multi_gpu_comparison_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 多GPU并行对比实验报告\n\n")
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 实验配置\n\n")
        f.write("| 参数 | 值 |\n")
        f.write("|------|----|\n")

        # 从第一个实验读取配置
        first_exp_dir = all_results[0]['exp_dir']
        with open(os.path.join(first_exp_dir, 'hyperparameters.json'), 'r') as hf:
            hp = json.load(hf)
            for key in ['num_layer', 'emb_dim', 'batch_size', 'num_epochs']:
                if key in hp:
                    f.write(f"| {key} | {hp[key]} |\n")

        f.write("\n## 所有实验结果\n\n")
        f.write("| 模型 | GPU | Test AUPR | Test AUC | Valid AUPR | Valid AUC | Best Epoch |\n")
        f.write("|------|-----|-----------|----------|------------|-----------|------------|\n")

        for result in all_results:
            f.write(f"| {result['exp_name']} | "
                   f"{result['gpu_id']} | "
                   f"{result['test_aupr']:.4f} | "
                   f"{result['test_auc']:.4f} | "
                   f"{result['valid_aupr']:.4f} | "
                   f"{result['valid_auc']:.4f} | "
                   f"{result['best_epoch']} |\n")

        f.write("\n## 按GNN类型分组对比\n\n")

        # 按GNN类型分组
        gnn_types = sorted(set([r['exp_name'].split('_')[0] for r in all_results]))

        for gnn_type in gnn_types:
            type_results = [r for r in all_results if r['exp_name'].startswith(gnn_type)]

            f.write(f"### {gnn_type.upper()}\n\n")
            f.write("| 模型 | Test AUPR | Test AUC | 提升 |\n")
            f.write("|------|-----------|----------|------|\n")

            baseline = next((r for r in type_results if 'baseline' in r['exp_name'].lower()), None)

            for result in type_results:
                if baseline and result != baseline:
                    aupr_improve = (result['test_aupr'] - baseline['test_aupr']) / baseline['test_aupr'] * 100
                    improve_str = f"{aupr_improve:+.2f}%"
                else:
                    improve_str = "-"

                f.write(f"| {result['exp_name']} | "
                       f"{result['test_aupr']:.4f} | "
                       f"{result['test_auc']:.4f} | "
                       f"{improve_str} |\n")

            f.write("\n")

        f.write("## 总结\n\n")

        # 找到最佳模型
        best_result = max(all_results, key=lambda x: x['test_aupr'])
        f.write(f"- **最佳模型**: {best_result['exp_name']}\n")
        f.write(f"- **Test AUPR**: {best_result['test_aupr']:.4f}\n")
        f.write(f"- **Test AUC**: {best_result['test_auc']:.4f}\n\n")

        # 计算平均提升
        improvements = []
        for gnn_type in gnn_types:
            type_results = [r for r in all_results if r['exp_name'].startswith(gnn_type)]
            baseline = next((r for r in type_results if 'baseline' in r['exp_name'].lower()), None)
            kagnn = next((r for r in type_results if 'kagnn' in r['exp_name'].lower()), None)

            if baseline and kagnn:
                aupr_improve = (kagnn['test_aupr'] - baseline['test_aupr']) / baseline['test_aupr'] * 100
                improvements.append(aupr_improve)

        if improvements:
            avg_improve = sum(improvements) / len(improvements)
            f.write(f"- **平均AUPR提升**: {avg_improve:+.2f}%\n")

    print(f"\n详细报告已保存到: {report_path}")


def main():
    args = parse_args()

    # 设置日志输出到当前目录
    import sys
    log_file = open('training.log', 'w', buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    print("="*80)
    print("多GPU并行对比实验")
    print("="*80)

    # 解析GPU列表
    gpu_list = [int(x.strip()) for x in args.gpus.split(',')]
    print(f"\n使用GPU: {gpu_list}")

    # 解析GNN类型列表
    gnn_types = [x.strip() for x in args.gnn_types.split(',')]
    print(f"GNN类型: {gnn_types}")
    print(f"KAN类型: {args.kan_type}")
    print(f"训练轮数: {args.epochs}")

    # 数据集路径
    processed_dir = '/home/bioinfo202200130116/bioinfo/codebasev5/graphnn/data/processed'
    score = 'loewe_thresh'
    score_val = 1
    DSdataset_name = f'DrugComb_{score}_{score_val}'
    data_fname = 'data_v1'

    targetdata_dir = os.path.join(processed_dir, DSdataset_name, data_fname)
    targetdata_dir_exp = create_directory(os.path.join(targetdata_dir, "experiments"))

    # 创建总实验目录
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    main_exp_dir = create_directory(
        os.path.join(targetdata_dir_exp, f"multi_gpu_comparison_{time_stamp}")
    )

    print(f"\n实验目录: {main_exp_dir}")

    # 加载数据集
    print("\n加载数据集...")
    dataset = MoleculeDataset(root=targetdata_dir, dataset='tdcSynergy')
    print(f"数据集大小: {len(dataset)}")

    # 创建数据分区
    fold_partitions = get_stratified_partitions(
        dataset.data.y,
        num_folds=5,
        valid_set_portion=0.1,
        random_state=42
    )
    partition = fold_partitions[0]

    print(f"训练集: {len(partition['train'])}")
    print(f"验证集: {len(partition['validation'])}")
    print(f"测试集: {len(partition['test'])}")

    # 准备所有实验
    experiments = []
    gpu_idx = 0

    # 解析use_kan和use_hypergraph参数
    use_kan_options = []
    if args.use_kan == 'true':
        use_kan_options = [True]
    elif args.use_kan == 'false':
        use_kan_options = [False]
    else:  # 'both'
        use_kan_options = [False, True]

    use_hypergraph_options = []
    if args.use_hypergraph == 'true':
        use_hypergraph_options = [True]
    elif args.use_hypergraph == 'false':
        use_hypergraph_options = [False]
    else:  # 'both'
        use_hypergraph_options = [False, True]

    # 为每种GNN类型创建实验
    for gnn_type in gnn_types:
        for use_kan in use_kan_options:
            for use_hypergraph in use_hypergraph_options:
                # 创建训练参数
                tp = create_training_params(args, gnn_type, use_kan, use_hypergraph)

                # 生成实验名称
                exp_name_parts = [gnn_type.upper()]
                if use_kan:
                    exp_name_parts.append(f"KAGNN_{args.kan_type}")
                else:
                    exp_name_parts.append("Baseline")
                if use_hypergraph:
                    exp_name_parts.append("Hypergraph")
                else:
                    exp_name_parts.append("NoHypergraph")

                exp_name = "_".join(exp_name_parts)

                # 创建实验目录
                exp_dir = create_directory(os.path.join(main_exp_dir, exp_name))
                create_directory(os.path.join(exp_dir, "predictions"))
                create_directory(os.path.join(exp_dir, "modelstates"))

                # 添加到实验列表
                experiments.append((
                    targetdata_dir,
                    partition,
                    tp,
                    exp_name,
                    exp_dir,
                    gpu_list[gpu_idx % len(gpu_list)]
                ))
                gpu_idx += 1

    print(f"\n总共 {len(experiments)} 个实验将在 {len(gpu_list)} 个GPU上运行")
    print("\n实验列表:")
    for i, (_, _, tp, exp_name, _, gpu_id) in enumerate(experiments):
        kan_status = "✓" if tp['use_kan'] else "✗"
        hg_status = "✓" if tp['use_hypergraph'] else "✗"
        print(f"  {i+1}. {exp_name:<40} (GPU {gpu_id}) [KAN:{kan_status} HG:{hg_status}]")

    # 并行运行所有实验
    print("\n开始并行训练...")
    print("="*80)

    with mp.Pool(processes=len(gpu_list)) as pool:
        results = pool.map(run_single_experiment_wrapper, experiments)

    # 收集结果
    all_results = []
    all_curves = {}

    for result, curves in results:
        if result is not None:
            all_results.append(result)
            all_curves[result['exp_name']] = curves

    if not all_results:
        print("\n所有实验都失败了!")
        return

    print("\n" + "="*80)
    print("所有实验完成!")
    print("="*80)

    # 生成报告和图表
    print("\n生成对比报告和图表...")

    # 按GNN类型分组
    results_by_type = {}
    for result in all_results:
        gnn_type = result['exp_name'].split('_')[0]
        if gnn_type not in results_by_type:
            results_by_type[gnn_type] = []
        results_by_type[gnn_type].append(result)

    # 为每种GNN类型绘制对比图
    plot_comparison_by_gnn_type(results_by_type, main_exp_dir)

    # 绘制总体对比图
    plot_overall_comparison(all_results, main_exp_dir)

    # 生成详细报告
    generate_report(all_results, main_exp_dir)

    # 保存结果CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(main_exp_dir, 'all_results.csv'), index=False)

    print(f"\n结果保存在: {main_exp_dir}")
    print("\n生成的文件:")
    print("  - all_results.csv: 所有结果数据")
    print("  - multi_gpu_comparison_report.md: 详细报告")
    print("  - overall_comparison.png: 总体对比图")
    for gnn_type in results_by_type.keys():
        print(f"  - comparison_{gnn_type}.png: {gnn_type.upper()}对比图")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
