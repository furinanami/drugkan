"""
训练函数 - 支持GNN/KAGNN切换
在原始train_functions的基础上添加KAN支持
"""
import os
import sys
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

import deepadr
from deepadr.dataset import *
from deepadr.utilities import *
from deepadr.chemfeatures import *

# 导入原始模型
from deepadr.model_gnn_ogb import DeepAdr_SiameseTrf, ExpressionNN, DeepSynergy
from deepadr.model_attn_siamese import GeneEmbAttention, GeneEmbProjAttention

# 导入KAGNN模型
from deepadr.model_kagnn import KAGNN

from ogb.graphproppred import Evaluator

import json
import functools

fdtype = torch.float32

torch.set_printoptions(precision=6)


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def F_score(a, b):
    return (2 * a * b) / (a + b)


def generate_tp_hp(tp, hp, hp_names):
    tphp = deepcopy(tp)
    for i, n in enumerate(hp_names):
        tphp[n] = hp[i]
    return tphp


def build_predictions_df(ids, true_class, pred_class, prob_scores):
    prob_scores_dict = {}
    for i in range(prob_scores.shape[-1]):
        prob_scores_dict[f'prob_score_class{i}'] = prob_scores[:, i]

    df_dict = {
        'id': ids,
        'true_class': true_class,
        'pred_class': pred_class
    }
    df_dict.update(prob_scores_dict)
    predictions_df = pd.DataFrame(df_dict)
    predictions_df.set_index('id', inplace=True)
    return predictions_df


def run_exp_kagnn(queue, used_dataset, gpu_num, tp, exp_dir, partition):
    """
    运行KAGNN实验

    核心参数：
        tp['use_kan']: bool, 是否使用KAN增强（默认False）
        tp['kan_type']: str, KAN类型 ('fourier' 或 'bspline'，默认'fourier')

    其他参数与原始run_exp相同
    """

    num_classes = 2

    targetdata_dir_raw = os.path.abspath(exp_dir + "/../../raw")
    targetdata_dir_processed = os.path.abspath(exp_dir + "/../../processed")

    device_gpu = get_device(True, index=gpu_num)
    print("gpu:", device_gpu)

    # 序列化超参数
    json.dump(tp, open(exp_dir + "/hyperparameters.json", 'w'))

    tp['nonlin_func'] = torch.nn.ReLU()

    # 获取KAN配置（如果未指定则使用默认值）
    use_kan = tp.get('use_kan', False)
    kan_type = tp.get('kan_type', 'fourier')

    print(f"\n{'='*60}")
    print(f"模型配置:")
    print(f"  GNN类型: {tp['gnn_type']}")
    print(f"  使用KAN: {use_kan}")
    if use_kan:
        print(f"  KAN类型: {kan_type}")
    print(f"{'='*60}\n")

    # 创建数据集
    train_dataset = Subset(used_dataset, partition['train'])
    val_dataset = Subset(used_dataset, partition['validation'])
    test_dataset = Subset(used_dataset, partition['test'])

    train_loader = DataLoader(train_dataset, batch_size=tp["batch_size"],
                            shuffle=True, follow_batch=['x_a', 'x_b'])
    valid_loader = DataLoader(val_dataset, batch_size=tp["batch_size"],
                            shuffle=False, follow_batch=['x_a', 'x_b'])
    test_loader = DataLoader(test_dataset, batch_size=tp["batch_size"],
                           shuffle=False, follow_batch=['x_a', 'x_b'])

    loaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}

    # 创建GNN模型（支持KAN增强）
    gnn_model = KAGNN(
        gnn_type=tp["gnn_type"],
        num_layer=tp["num_layer"],
        emb_dim=tp["emb_dim"],
        drop_ratio=0.5,
        JK="last",
        graph_pooling=tp["graph_pooling"],
        virtual_node=False,
        with_edge_attr=False,
        use_kan=use_kan,
        kan_type=kan_type
    ).to(device=device_gpu, dtype=fdtype)

    # Expression模型（保持不变）
    expression_model = DeepSynergy(
        D_in=(2 * tp["emb_dim"]) + tp["expression_input_size"],
        H1=tp['exp_H1'],
        H2=tp['exp_H2'],
        drop=tp['p_dropout']
    ).to(device=device_gpu, dtype=fdtype)

    models_param = list(gnn_model.parameters()) + list(expression_model.parameters())

    model_name = "kagnn" if use_kan else "ogb"
    models = [
        (gnn_model, f'{model_name}_GNN'),
        (expression_model, f'{model_name}_Expression')
    ]

    # 计算类别权重
    y_weights = compute_class_weights(used_dataset.data.y[partition['train']])
    class_weights = torch.tensor(y_weights).type(fdtype).to(device_gpu)

    # 优化器和学习率调度器
    num_iter = len(train_loader)
    c_step_size = int(np.ceil(5 * num_iter))

    base_lr = tp['base_lr']
    max_lr = tp['max_lr_mul'] * base_lr
    optimizer = torch.optim.Adam(models_param, weight_decay=tp["l2_reg"], lr=base_lr)
    cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr, max_lr, step_size_up=c_step_size,
        mode='triangular', cycle_momentum=False
    )

    # 损失函数
    loss_nlll = torch.nn.NLLLoss(weight=class_weights, reduction='mean')
    loss_contrastive = ContrastiveLoss(0.5, reduction='mean')

    # 记录训练曲线
    valid_curve_aupr = []
    test_curve_aupr = []
    train_curve_aupr = []

    valid_curve_auc = []
    test_curve_auc = []
    train_curve_auc = []

    best_fscore = 0
    best_epoch = 0

    # 训练循环
    for epoch in range(tp["num_epochs"]):
        print(f"\n=====Epoch {epoch}/{tp['num_epochs']}=====")
        print('Training...')

        for m, m_name in models:
            m.train()

        for i_batch, batch in enumerate(train_loader):
            batch = batch.to(device_gpu)

            # Drug A和Drug B的GNN编码
            h_a = gnn_model(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
            h_b = gnn_model(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)

            # Expression特征
            h_e = batch.expression.type(fdtype)

            # 拼接特征
            triplet = torch.cat([h_a, h_b, h_e], axis=-1)

            # 预测
            logsoftmax_scores = expression_model(triplet)

            # 计算损失
            loss = loss_nlll(logsoftmax_scores, batch.y.type(torch.long))
            loss.backward()

            # 修复2: 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(models_param, max_norm=1.0)

            optimizer.step()
            cyc_scheduler.step()
            optimizer.zero_grad()

        print('Evaluating...')

        perfs = {}

        for dsettype in ["train", "test", "valid"]:
            for m, m_name in models:
                m.eval()

            pred_class = []
            ref_class = []
            prob_scores = []
            l_ids = []

            for i_batch, batch in enumerate(loaders[dsettype]):
                batch = batch.to(device_gpu)

                h_a = gnn_model(batch.x_a, batch.edge_index_a, batch.edge_attr_a, batch.x_a_batch)
                h_b = gnn_model(batch.x_b, batch.edge_index_b, batch.edge_attr_b, batch.x_b_batch)

                h_e = batch.expression.type(fdtype)

                triplet = torch.cat([h_a, h_b, h_e], axis=-1)

                logsoftmax_scores = expression_model(triplet)

                __, y_pred_clss = torch.max(logsoftmax_scores, -1)

                y_pred_prob = torch.exp(logsoftmax_scores.detach().cpu()).numpy()

                pred_class.extend(y_pred_clss.view(-1).tolist())
                ref_class.extend(batch.y.view(-1).tolist())
                prob_scores.append(y_pred_prob)
                l_ids.extend(batch.id.view(-1).tolist())

            prob_scores_arr = np.concatenate(prob_scores, axis=0)

            dset_perf = perfmetric_report(
                pred_class, ref_class, prob_scores_arr[:, 1], epoch,
                outlog=os.path.join(exp_dir, dsettype + ".log")
            )

            perfs[dsettype] = dset_perf

            if dsettype == "test":
                fscore = F_score(perfs['test'].s_aupr, perfs['test'].s_auc)
                if fscore > best_fscore:
                    best_fscore = fscore
                    best_epoch = epoch

                    # 保存最佳模型
                    torch.save({
                        'epoch': epoch,
                        'gnn_model_state_dict': gnn_model.state_dict(),
                        'expression_model_state_dict': expression_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_fscore': best_fscore,
                    }, os.path.join(exp_dir, 'modelstates', 'best_model.pt'))

                predictions_df = build_predictions_df(l_ids, ref_class, pred_class, prob_scores_arr)
                predictions_df.to_csv(
                    os.path.join(exp_dir, 'predictions', f'epoch_{epoch}_predictions_{dsettype}.csv')
                )

        print(f"Epoch {epoch}: Train AUPR={perfs['train'].s_aupr:.4f}, Valid AUPR={perfs['valid'].s_aupr:.4f}, Test AUPR={perfs['test'].s_aupr:.4f}")

        train_curve_aupr.append(perfs['train'].s_aupr)
        valid_curve_aupr.append(perfs['valid'].s_aupr)
        test_curve_aupr.append(perfs['test'].s_aupr)

        train_curve_auc.append(perfs['train'].s_auc)
        valid_curve_auc.append(perfs['valid'].s_auc)
        test_curve_auc.append(perfs['test'].s_auc)

    print('Finished training!')
    print(f'Best epoch: {best_epoch}, Best F-score: {best_fscore:.4f}')

    # 保存训练曲线
    df_curves = pd.DataFrame(np.array([
        train_curve_aupr, valid_curve_aupr, test_curve_aupr,
        train_curve_auc, valid_curve_auc, test_curve_auc
    ]).T)
    df_curves.columns = ['train_aupr', 'valid_aupr', 'test_aupr',
                        'train_auc', 'valid_auc', 'test_auc']
    df_curves.index.name = "epoch"
    df_curves.to_csv(exp_dir + "/curves.csv")

    # 绘制曲线
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_curve_aupr, label='Train')
    plt.plot(valid_curve_aupr, label='Valid')
    plt.plot(test_curve_aupr, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('AUPR')
    plt.legend()
    plt.title('AUPR Curves')

    plt.subplot(1, 2, 2)
    plt.plot(train_curve_auc, label='Train')
    plt.plot(valid_curve_auc, label='Valid')
    plt.plot(test_curve_auc, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('AUC Curves')

    plt.tight_layout()
    plt.savefig(exp_dir + "/curves.png", dpi=150)
    plt.close()

    queue.put(gpu_num)
