#!/usr/bin/env python3
"""
GNN vs KAGNN 对比实验脚本

功能：
1. 同时运行原版GNN和KAN增强版KAGNN
2. 对比两种模型在相同数据集上的性能
3. 支持傅里叶和B样条两种KAN实现
4. 生成详细的对比报告

使用方法：
    python run_comparison_experiments.py --gpu 1 --epochs 100
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

sys.path.insert(0, '/home/bioinfo202200130116/bioinfo/codebasev5/graphnn')

from deepadr.dataset import MoleculeDataset, get_stratified_partitions
from deepadr.utilities import create_directory
from deepadr.train_functions_kagnn import run_exp_kagnn


def parse_args():
    parser = argparse.ArgumentParser(description='GNN vs KAGNN Comparison Experiments')

    parser.add_argument('--gpu', type=int, default=1,
                       help='GPU device ID (default: 1)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=300,
                       help='Batch size (default: 300)')
    parser.add_argument('--gnn_type', type=str, default='gatv2',
                       choices=['gat', 'gatv2', 'gcn', 'gin'],
                       help='GNN type (default: gatv2)')
    parser.add_argument('--emb_dim', type=int, default=100,
                       help='Embedding dimension (default: 100)')
    parser.add_argument('--num_layer', type=int, default=5,
                       help='Number of GNN layers (default: 5)')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline GNN training')
    parser.add_argument('--skip_fourier', action='store_true',
                       help='Skip Fourier KAN training')
    parser.add_argument('--skip_bspline', action='store_true',
                       help='Skip B-spline KAN training')

    return parser.parse_args()


def create_training_params(args):
    """创建训练参数字典"""
    tp = {
        "batch_size": args.batch_size,
        "num_epochs": args.epochs,

        "emb_dim": args.emb_dim,
        "gnn_type": args.gnn_type,
        "num_layer": args.num_layer,
        "graph_pooling": "mean",

        "num_attn_heads": 2,
        "num_transformer_units": 1,
        "p_dropout": 0.3,
        "mlp_embed_factor": 2,
        "pooling_mode": 'attn',
        "dist_opt": 'cosine',

        "base_lr": 3e-4,
        "max_lr_mul": 10,
        "l2_reg": 1e-7,
        "loss_w": 1.,
        "margin_v": 1.,

        "expression_dim": 64,
        "expression_input_size": 946,
        "exp_H1": 4096,
        "exp_H2": 1024
    }
    return tp


def run_single_experiment(dataset, partition, tp, exp_name, exp_base_dir, gpu_num):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"开始实验: {exp_name}")
    print(f"{'='*80}\n")

    # 创建实验目录
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    exp_dir = create_directory(os.path.join(exp_base_dir, f"{exp_name}_{time_stamp}"))
    create_directory(os.path.join(exp_dir, "predictions"))
    create_directory(os.path.join(exp_dir, "modelstates"))

    # 运行实验
    queue = mp.Queue()
    run_exp_kagnn(queue, dataset, gpu_num, tp, exp_dir, partition)

    # 读取结果
    curves_df = pd.read_csv(os.path.join(exp_dir, "curves.csv"), index_col=0)

    # 获取最佳性能
    best_epoch = curves_df['test_aupr'].idxmax()
    best_results = {
        'exp_name': exp_name,
        'exp_dir': exp_dir,
        'best_epoch': best_epoch,
        'test_aupr': curves_df.loc[best_epoch, 'test_aupr'],
        'test_auc': curves_df.loc[best_epoch, 'test_auc'],
        'valid_aupr': curves_df.loc[best_epoch, 'valid_aupr'],
        'valid_auc': curves_df.loc[best_epoch, 'valid_auc'],
        'train_aupr': curves_df.loc[best_epoch, 'train_aupr'],
        'train_auc': curves_df.loc[best_epoch, 'train_auc'],
    }

    print(f"\n实验 {exp_name} 完成!")
    print(f"最佳epoch: {best_epoch}")
    print(f"Test AUPR: {best_results['test_aupr']:.4f}")
    print(f"Test AUC: {best_results['test_auc']:.4f}")

    return best_results, curves_df


def plot_comparison(all_results, all_curves, output_dir):
    """绘制对比图表"""

    # 1. 性能对比柱状图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    metrics = ['test_aupr', 'test_auc', 'valid_aupr', 'valid_auc']
    titles = ['Test AUPR', 'Test AUC', 'Validation AUPR', 'Validation AUC']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        exp_names = [r['exp_name'] for r in all_results]
        values = [r[metric] for r in all_results]

        bars = ax.bar(range(len(exp_names)), values, alpha=0.8)

        # 为最佳结果添加颜色
        best_idx = values.index(max(values))
        bars[best_idx].set_color('green')
        bars[best_idx].set_alpha(1.0)

        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Comparison')
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=150)
    plt.close()

    # 2. 训练曲线对比
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    curve_metrics = [
        ('test_aupr', 'Test AUPR'),
        ('test_auc', 'Test AUC'),
        ('valid_aupr', 'Validation AUPR'),
        ('valid_auc', 'Validation AUC')
    ]

    for idx, (metric, title) in enumerate(curve_metrics):
        ax = axes[idx // 2, idx % 2]

        for exp_name, curves_df in all_curves.items():
            ax.plot(curves_df.index, curves_df[metric], label=exp_name, linewidth=2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'{title} Training Curves')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves_comparison.png'), dpi=150)
    plt.close()

    print(f"\n对比图表已保存到: {output_dir}")


def generate_report(all_results, output_dir):
    """生成详细的对比报告"""

    # 创建结果DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index('exp_name')

    # 保存为CSV
    results_df.to_csv(os.path.join(output_dir, 'comparison_results.csv'))

    # 生成Markdown报告
    report_path = os.path.join(output_dir, 'comparison_report.md')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# GNN vs KAGNN 对比实验报告\n\n")
        f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 实验配置\n\n")
        f.write("| 参数 | 值 |\n")
        f.write("|------|----|\n")

        # 从第一个实验读取配置
        first_exp_dir = all_results[0]['exp_dir']
        with open(os.path.join(first_exp_dir, 'hyperparameters.json'), 'r') as hf:
            hp = json.load(hf)
            for key in ['gnn_type', 'num_layer', 'emb_dim', 'batch_size', 'num_epochs']:
                if key in hp:
                    f.write(f"| {key} | {hp[key]} |\n")

        f.write("\n## 性能对比\n\n")
        f.write("### Test Set 性能\n\n")
        f.write("| 模型 | AUPR | AUC | Best Epoch |\n")
        f.write("|------|------|-----|------------|\n")

        for result in all_results:
            f.write(f"| {result['exp_name']} | "
                   f"{result['test_aupr']:.4f} | "
                   f"{result['test_auc']:.4f} | "
                   f"{result['best_epoch']} |\n")

        f.write("\n### Validation Set 性能\n\n")
        f.write("| 模型 | AUPR | AUC |\n")
        f.write("|------|------|-----|\n")

        for result in all_results:
            f.write(f"| {result['exp_name']} | "
                   f"{result['valid_aupr']:.4f} | "
                   f"{result['valid_auc']:.4f} |\n")

        f.write("\n## 性能提升分析\n\n")

        # 找到baseline
        baseline = None
        for result in all_results:
            if 'baseline' in result['exp_name'].lower():
                baseline = result
                break

        if baseline:
            f.write("相对于Baseline GNN的性能提升:\n\n")
            f.write("| 模型 | AUPR提升 | AUC提升 |\n")
            f.write("|------|----------|----------|\n")

            for result in all_results:
                if result['exp_name'] != baseline['exp_name']:
                    aupr_improve = (result['test_aupr'] - baseline['test_aupr']) / baseline['test_aupr'] * 100
                    auc_improve = (result['test_auc'] - baseline['test_auc']) / baseline['test_auc'] * 100
                    f.write(f"| {result['exp_name']} | "
                           f"{aupr_improve:+.2f}% | "
                           f"{auc_improve:+.2f}% |\n")

        f.write("\n## 结论\n\n")

        # 找到最佳模型
        best_result = max(all_results, key=lambda x: x['test_aupr'])
        f.write(f"- 最佳模型: **{best_result['exp_name']}**\n")
        f.write(f"- Test AUPR: {best_result['test_aupr']:.4f}\n")
        f.write(f"- Test AUC: {best_result['test_auc']:.4f}\n")

        if baseline and best_result['exp_name'] != baseline['exp_name']:
            aupr_improve = (best_result['test_aupr'] - baseline['test_aupr']) / baseline['test_aupr'] * 100
            f.write(f"- 相对Baseline提升: {aupr_improve:+.2f}%\n")

    print(f"\n详细报告已保存到: {report_path}")


def main():
    args = parse_args()

    print("="*80)
    print("GNN vs KAGNN 对比实验")
    print("="*80)
    print(f"\n配置:")
    print(f"  GPU: {args.gpu}")
    print(f"  Epochs: {args.epochs}")
    print(f"  GNN类型: {args.gnn_type}")
    print(f"  嵌入维度: {args.emb_dim}")
    print(f"  GNN层数: {args.num_layer}")

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # 数据集路径
    processed_dir = '/home/bioinfo202200130116/bioinfo/codebasev5/graphnn/data/processed'
    score = 'loewe_thresh'
    score_val = 1
    DSdataset_name = f'DrugComb_{score}_{score_val}'
    data_fname = 'data_v1'

    targetdata_dir = os.path.join(processed_dir, DSdataset_name, data_fname)
    targetdata_dir_exp = create_directory(os.path.join(targetdata_dir, "experiments"))

    # 创建对比实验目录
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    comparison_dir = create_directory(
        os.path.join(targetdata_dir_exp, f"comparison_{time_stamp}")
    )

    print(f"\n对比实验目录: {comparison_dir}")

    # 加载数据集
    print("\n加载数据集...")
    dataset = MoleculeDataset(root=targetdata_dir)
    print(f"数据集大小: {len(dataset)}")

    # 创建数据分区（使用相同的随机种子确保公平对比）
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

    # 创建基础训练参数
    base_tp = create_training_params(args)

    # 存储所有实验结果
    all_results = []
    all_curves = {}

    # 实验1: Baseline GNN (不使用KAN)
    if not args.skip_baseline:
        tp_baseline = base_tp.copy()
        tp_baseline['use_kan'] = False

        result, curves = run_single_experiment(
            dataset, partition, tp_baseline,
            "Baseline_GNN", comparison_dir, 0
        )
        all_results.append(result)
        all_curves['Baseline_GNN'] = curves

    # 实验2: KAGNN with Fourier KAN
    if not args.skip_fourier:
        tp_fourier = base_tp.copy()
        tp_fourier['use_kan'] = True
        tp_fourier['kan_type'] = 'fourier'

        result, curves = run_single_experiment(
            dataset, partition, tp_fourier,
            "KAGNN_Fourier", comparison_dir, 0
        )
        all_results.append(result)
        all_curves['KAGNN_Fourier'] = curves

    # 实验3: KAGNN with B-spline KAN
    if not args.skip_bspline:
        tp_bspline = base_tp.copy()
        tp_bspline['use_kan'] = True
        tp_bspline['kan_type'] = 'bspline'

        result, curves = run_single_experiment(
            dataset, partition, tp_bspline,
            "KAGNN_BSpline", comparison_dir, 0
        )
        all_results.append(result)
        all_curves['KAGNN_BSpline'] = curves

    # 生成对比报告和图表
    print("\n生成对比报告...")
    plot_comparison(all_results, all_curves, comparison_dir)
    generate_report(all_results, comparison_dir)

    print("\n" + "="*80)
    print("所有实验完成!")
    print("="*80)
    print(f"\n结果保存在: {comparison_dir}")
    print("\n生成的文件:")
    print("  - comparison_results.csv: 详细结果数据")
    print("  - comparison_report.md: Markdown格式报告")
    print("  - performance_comparison.png: 性能对比柱状图")
    print("  - training_curves_comparison.png: 训练曲线对比")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
