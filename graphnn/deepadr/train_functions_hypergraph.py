"""
训练函数：DeepDDS with Hypergraph Integration
支持开关控制是否使用超图
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
from deepadr.model_hypergraph import (
    DeepDDS_Hypergraph,
    DeepDDS_Hypergraph_WithReconstruction
)
from ogb.graphproppred import Evaluator

import json
import functools

fdtype = torch.float32

torch.set_printoptions(precision=6)


def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def F_score(a, b):
    return (2*a*b)/(a+b)


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


def run_exp_deepdds_hypergraph(queue, used_dataset, gpu_num, tp, exp_dir, partition):
    """
    运行DeepDDS超图实验

    Args:
        queue: 多进程队列
        used_dataset: 数据集
        gpu_num: GPU编号
        tp: 超参数字典，必须包含：
            - use_hypergraph: bool, 是否使用超图（True=超图模式，False=简单MLP模式）
            - use_reconstruction: bool, 是否使用重构损失（仅use_hypergraph=True时有效）
            - alpha: float, 重构损失权重（仅use_reconstruction=True时有效）
            - 其他标准超参数...
        exp_dir: 实验目录
        partition: 数据划分字典
    """

    num_classes = 2

    targetdata_dir_raw = os.path.abspath(exp_dir + "/../../raw")
    targetdata_dir_processed = os.path.abspath(exp_dir + "/../../processed")

    device_gpu = get_device(True, index=gpu_num)
    print("gpu:", device_gpu)
    print(f"使用超图模式: {tp.get('use_hypergraph', True)}")
    print(f"使用重构损失: {tp.get('use_reconstruction', False)}")

    # 保存超参数
    json.dump(tp, open(exp_dir + "/hyperparameters.json", 'w'))

    tp['nonlin_func'] = nn.ReLU()

    # 数据标准化
    expression_scaler = TorchStandardScaler()
    expression_scaler.fit(used_dataset.data.expression[partition['train']])

    # 数据加载器
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

    # 构建模型
    use_hypergraph = tp.get('use_hypergraph', True)
    use_reconstruction = tp.get('use_reconstruction', False) and use_hypergraph

    model_kwargs = {
        'num_features_xd': 9,
        'gat_output_dim': tp["emb_dim"],
        'expression_input_size': tp["expression_input_size"],
        'exp_H1': tp['exp_H1'],
        'exp_H2': tp['exp_H2'],
        'unified_dim': tp.get('unified_dim', 256),
        'hypergraph_hidden': tp.get('hypergraph_hidden', 512),
        'decoder_hidden1': tp.get('decoder_hidden1', 512),
        'decoder_hidden2': tp.get('decoder_hidden2', 256),
        'num_classes': num_classes,
        'dropout': tp['p_dropout'],
        'num_attn_heads': tp["num_attn_heads"],
        'use_hypergraph': use_hypergraph
    }

    if use_reconstruction:
        hypergraph_model = DeepDDS_Hypergraph_WithReconstruction(**model_kwargs)
    else:
        hypergraph_model = DeepDDS_Hypergraph(**model_kwargs)

    hypergraph_model = hypergraph_model.to(device=device_gpu, dtype=fdtype)

    model_name = "hypergraph" if use_hypergraph else "mlp"
    models = [(hypergraph_model, f'{model_name}_model')]

    # 计算类别权重
    y_weights = compute_class_weights(used_dataset.data.y[partition['train']])
    class_weights = torch.tensor(y_weights).type(fdtype).to(device_gpu)

    # 优化器和学习率调度器
    num_iter = len(train_loader)
    c_step_size = int(np.ceil(5*num_iter))

    base_lr = tp['base_lr']
    max_lr = tp['max_lr_mul']*base_lr
    optimizer = torch.optim.Adam(hypergraph_model.parameters(),
                                weight_decay=tp["l2_reg"], lr=base_lr)
    cyc_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr, max_lr, step_size_up=c_step_size,
        mode='triangular', cycle_momentum=False
    )

    # 损失函数
    loss_nlll = torch.nn.NLLLoss(weight=class_weights, reduction='mean')
    loss_mse = torch.nn.MSELoss(reduction='mean')

    # 训练曲线记录
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
        print("=====Epoch {}".format(epoch))
        print('Training...')

        hypergraph_model.train()

        for i_batch, batch in enumerate(train_loader):
            batch = batch.to(device_gpu)

            # 确保节点特征是long类型（用于embedding）
            batch.x_a = batch.x_a.long()
            batch.x_b = batch.x_b.long()

            if use_reconstruction:
                # 带重构损失的训练
                logsoftmax_scores, rec_drug, rec_cline = hypergraph_model(batch)

                loss_main = loss_nlll(logsoftmax_scores, batch.y.type(torch.long))

                if rec_drug is not None and rec_cline is not None:
                    # 计算重构损失（需要提供真实的相似度矩阵）
                    # 这里简化处理，实际使用时需要传入真实的相似度矩阵
                    loss_rec = 0  # 占位符
                    alpha = tp.get('alpha', 0.4)
                    loss = (1 - alpha) * loss_main + alpha * loss_rec
                else:
                    loss = loss_main
            else:
                # 不使用重构损失的训练
                logsoftmax_scores = hypergraph_model(batch)
                loss = loss_nlll(logsoftmax_scores, batch.y.type(torch.long))

            loss.backward()
            optimizer.step()
            cyc_scheduler.step()
            optimizer.zero_grad()

        print('Evaluating...')

        perfs = {}

        for dsettype in ["train", "test", "valid"]:
            hypergraph_model.eval()

            pred_class = []
            ref_class = []
            prob_scores = []
            l_ids = []

            for i_batch, batch in enumerate(loaders[dsettype]):
                batch = batch.to(device_gpu)

                # 确保节点特征是long类型（用于embedding）
                batch.x_a = batch.x_a.long()
                batch.x_b = batch.x_b.long()

                with torch.no_grad():
                    if use_reconstruction:
                        logsoftmax_scores, _, _ = hypergraph_model(batch)
                    else:
                        logsoftmax_scores = hypergraph_model(batch)

                __, y_pred_clss = torch.max(logsoftmax_scores, -1)
                y_pred_prob = torch.exp(logsoftmax_scores.detach().cpu()).numpy()

                pred_class.extend(y_pred_clss.view(-1).tolist())
                ref_class.extend(batch.y.view(-1).tolist())
                prob_scores.append(y_pred_prob)
                l_ids.extend(batch.id.view(-1).tolist())

            prob_scores_arr = np.concatenate(prob_scores, axis=0)

            dset_perf = perfmetric_report(pred_class, ref_class, prob_scores_arr[:,1], epoch,
                                          outlog=os.path.join(exp_dir, dsettype + ".log"))

            perfs[dsettype] = dset_perf

            if (dsettype == "test"):
                fscore = F_score(perfs['test'].s_aupr, perfs['test'].s_auc)
                if (fscore > best_fscore):
                    best_fscore = fscore
                    best_epoch = epoch

                predictions_df = build_predictions_df(l_ids, ref_class, pred_class, prob_scores_arr)
                predictions_df.to_csv(os.path.join(exp_dir, 'predictions',
                                                   f'epoch_{epoch}_predictions_{dsettype}.csv'))

        print({'Train': perfs['train'], 'Validation': perfs['valid'], 'Test': perfs['test']})

        train_curve_aupr.append(perfs['train'].s_aupr)
        valid_curve_aupr.append(perfs['valid'].s_aupr)
        test_curve_aupr.append(perfs['test'].s_aupr)

        train_curve_auc.append(perfs['train'].s_auc)
        valid_curve_auc.append(perfs['valid'].s_auc)
        test_curve_auc.append(perfs['test'].s_auc)

    print('Finished training!')
    print(f'Best epoch: {best_epoch}, Best F-score: {best_fscore:.4f}')

    # 保存训练曲线
    df_curves = pd.DataFrame(np.array([train_curve_aupr, valid_curve_aupr, test_curve_aupr,
                                       train_curve_auc, valid_curve_auc, test_curve_auc]).T)
    df_curves.columns = ['train_aupr', 'valid_aupr', 'test_aupr', 'train_auc', 'valid_auc', 'test_auc']
    df_curves.index.name = "epoch"
    df_curves.to_csv(exp_dir + "/curves.csv")
    sns.lineplot(data=df_curves).figure.savefig(exp_dir + "/curves.png")

    queue.put(gpu_num)
