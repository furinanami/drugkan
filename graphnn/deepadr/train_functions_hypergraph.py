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


def build_regression_predictions_df(ids, true_score, pred_score):
    predictions_df = pd.DataFrame({
        'id': ids,
        'true_score': true_score,
        'pred_score': pred_score,
    })
    predictions_df.set_index('id', inplace=True)
    return predictions_df


def get_regression_target(batch, target_score='loewe'):
    attr_name = f'{target_score}_score'
    if not hasattr(batch, attr_name):
        raise AttributeError(f"Regression target '{attr_name}' is missing from the batch.")
    return getattr(batch, attr_name).view(-1).type(fdtype)


def fit_regression_target_scaler(used_dataset, train_indices, target_score='loewe'):
    attr_name = f'{target_score}_score'
    if not hasattr(used_dataset.data, attr_name):
        raise AttributeError(f"Regression target '{attr_name}' is missing from the dataset.")

    train_targets = getattr(used_dataset.data, attr_name)[train_indices].view(-1).type(fdtype)
    target_mean = train_targets.mean()
    target_std = train_targets.std(unbiased=False).clamp(min=1e-6)
    return target_mean, target_std


def regression_metric_report(pred_scores, true_scores, epoch, outlog=None):
    pred_scores = np.asarray(pred_scores, dtype=np.float64).reshape(-1)
    true_scores = np.asarray(true_scores, dtype=np.float64).reshape(-1)
    diff = pred_scores - true_scores
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    denom = float(np.sum((true_scores - true_scores.mean()) ** 2))
    r2 = float(1.0 - np.sum(diff ** 2) / denom) if denom > 0 else 0.0
    pearson = (
        float(np.corrcoef(pred_scores, true_scores)[0, 1])
        if pred_scores.size > 1 and np.std(pred_scores) > 0 and np.std(true_scores) > 0
        else 0.0
    )
    spearman = float(
        pd.Series(pred_scores).rank().corr(pd.Series(true_scores).rank())
    ) if pred_scores.size > 1 else 0.0
    if np.isnan(spearman):
        spearman = 0.0

    score = ModelScore(epoch, r2, -rmse, mae, pearson, spearman)
    score.rmse = rmse
    score.mse = mse
    score.mae = mae
    score.r2 = r2
    score.pearson = pearson
    score.spearman = spearman

    if outlog is not None:
        with open(outlog, 'a') as f:
            f.write(
                f"epoch={epoch}, rmse={rmse:.6f}, mae={mae:.6f}, "
                f"r2={r2:.6f}, pearson={pearson:.6f}, spearman={spearman:.6f}\n"
            )
    return score


def run_exp_deepdds_hypergraph(queue, used_dataset, gpu_num, tp, exp_dir, partition):
    """
    运行DeepDDS超图实验

    Args:
        queue: 多进程队列
        used_dataset: 数据集
        gpu_num: GPU编号
        tp: 超参数字典，必须包含：
            - hypergraph_mode: str, 模式选择 ('hypergraph', 'kan_hypergraph', 'kan_aggregated')
            - use_reconstruction: bool, 是否使用重构损失（仅hypergraph模式有效）
            - alpha: float, 重构损失权重（仅use_reconstruction=True时有效）
            - use_kan: bool, KAN开关（用于kan_aggregated模式）
            - kan_type: str, KAN类型（'fourier', 'bspline'等）
            - 其他标准超参数...
        exp_dir: 实验目录
        partition: 数据划分字典
    """

    task = tp.get('task', 'classification')
    target_score = tp.get('target_score', 'loewe')
    num_classes = 1 if task == 'regression' else 2

    targetdata_dir_raw = os.path.abspath(exp_dir + "/../../raw")
    targetdata_dir_processed = os.path.abspath(exp_dir + "/../../processed")

    device_gpu = get_device(True, index=gpu_num)
    print("gpu:", device_gpu)
    print(f"超图模式: {tp.get('hypergraph_mode', 'hypergraph')}")
    print(f"任务类型: {task}")
    if task == 'regression':
        print(f"回归目标: {target_score}_score")
    print(f"使用重构损失: {tp.get('use_reconstruction', False)}")
    if tp.get('hypergraph_mode') in ['kan_hypergraph', 'kan_aggregated']:
        print(f"使用KAN: {tp.get('use_kan', True)}")
        print(f"KAN类型: {tp.get('kan_type', 'fourier')}")

    # 数据标准化
    expression_scaler = TorchStandardScaler()
    expression_scaler.fit(used_dataset.data.expression[partition['train']])

    standardize_regression_target = bool(tp.get('standardize_regression_target', True))
    if task == 'regression':
        target_mean, target_std = fit_regression_target_scaler(
            used_dataset, partition['train'], target_score
        )
        if not standardize_regression_target:
            target_mean = torch.tensor(0.0, dtype=fdtype)
            target_std = torch.tensor(1.0, dtype=fdtype)
        target_mean_value = float(target_mean.item())
        target_std_value = float(target_std.item())
        target_mean_gpu = target_mean.to(device_gpu)
        target_std_gpu = target_std.to(device_gpu)
        tp['standardize_regression_target'] = standardize_regression_target
        tp['target_score_mean'] = target_mean_value
        tp['target_score_std'] = target_std_value
        print(
            f"回归target标准化: enabled={standardize_regression_target}, "
            f"mean={target_mean_value:.6f}, std={target_std_value:.6f}"
        )
    else:
        target_mean_gpu = None
        target_std_gpu = None

    # 保存超参数。必须在添加nn.ReLU等不可JSON序列化对象之前保存。
    json.dump(tp, open(exp_dir + "/hyperparameters.json", 'w'))

    tp['nonlin_func'] = nn.ReLU()

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

    print("图结构构建: 仅使用当前batch。每行样本加一条边/超边；不检索额外context样本。")

    # 构建模型
    hypergraph_mode = tp.get('hypergraph_mode', 'hypergraph')
    use_reconstruction = tp.get('use_reconstruction', False) and (hypergraph_mode == 'hypergraph')

    model_kwargs = {
        'num_features_xd': 9,
        'gat_output_dim': tp["emb_dim"],
        'expression_input_size': tp["expression_input_size"],
        'exp_H1': tp['exp_H1'],
        'exp_H2': tp['exp_H2'],
        'unified_dim': tp.get('unified_dim', 128),
        'hypergraph_hidden': tp.get('hypergraph_hidden', 256),
        'decoder_hidden1': tp.get('decoder_hidden1', 256),
        'decoder_hidden2': tp.get('decoder_hidden2', 128),
        'num_classes': num_classes,
        'dropout': tp['p_dropout'],
        'num_attn_heads': tp["num_attn_heads"],
        'hypergraph_mode': hypergraph_mode,
        'use_kan': tp.get('use_hypergraph_kan', tp.get('use_kan', True)),
        'kan_type': tp.get('kan_type', 'fourier'),
        'use_drug_kan': tp.get('use_drug_kan', tp.get('use_kan', True)),
        'use_drug_readout_kan': tp.get('use_drug_readout_kan', False),
        'gnn_type': tp.get('gnn_type', 'gcn'),
        'num_layer': tp.get('num_layer', 5),
        'drug_jk': tp.get('drug_jk', 'last'),
        'graph_pooling': tp.get('graph_pooling', 'mean'),
        'decoder_type': tp.get('decoder_type', 'kan' if hypergraph_mode == 'kan_mlp' else 'mlp'),
        'task': task
    }

    if use_reconstruction:
        hypergraph_model = DeepDDS_Hypergraph_WithReconstruction(**model_kwargs)
    else:
        hypergraph_model = DeepDDS_Hypergraph(**model_kwargs)

    hypergraph_model = hypergraph_model.to(device=device_gpu, dtype=fdtype)

    model_name = hypergraph_mode
    models = [(hypergraph_model, f'{model_name}_model')]

    # 计算类别权重
    if task == 'classification':
        y_weights = compute_class_weights(used_dataset.data.y[partition['train']])
        class_weights = torch.tensor(y_weights).type(fdtype).to(device_gpu)
    else:
        class_weights = None

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
    loss_nlll = torch.nn.NLLLoss(weight=class_weights, reduction='mean') if task == 'classification' else None
    loss_mse = torch.nn.MSELoss(reduction='mean')

    # 训练曲线记录
    valid_curve_aupr = []
    test_curve_aupr = []
    train_curve_aupr = []

    valid_curve_auc = []
    test_curve_auc = []
    train_curve_auc = []

    train_curve_rmse = []
    valid_curve_rmse = []
    test_curve_rmse = []
    train_curve_mae = []
    valid_curve_mae = []
    test_curve_mae = []
    train_curve_r2 = []
    valid_curve_r2 = []
    test_curve_r2 = []

    best_fscore = 0
    best_epoch = 0
    best_valid_aupr = -np.inf
    best_valid_rmse = np.inf
    epochs_without_improvement = 0
    early_stopping_patience = int(tp.get('early_stopping_patience', 0) or 0)
    os.makedirs(os.path.join(exp_dir, 'modelstates'), exist_ok=True)
    best_model_path = os.path.join(exp_dir, 'modelstates', 'best_model.pt')

    def save_best_checkpoint(epoch, metric_name, metric_value):
        simple_hyperparameters = {
            k: v for k, v in tp.items()
            if isinstance(v, (str, int, float, bool, list, tuple, dict, type(None)))
        }
        torch.save({
            'epoch': epoch,
            'model_state_dict': hypergraph_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric_name': metric_name,
            'metric_value': float(metric_value),
            'hyperparameters': simple_hyperparameters,
        }, best_model_path)

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
            if task == 'regression':
                pred_scores = hypergraph_model(batch)
                target_scores_raw = get_regression_target(batch, target_score).to(device_gpu)
                target_scores = (target_scores_raw - target_mean_gpu) / target_std_gpu
                loss = loss_mse(pred_scores.view(-1), target_scores)
            elif use_reconstruction:
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
            pred_scores = []
            ref_scores = []
            l_ids = []

            for i_batch, batch in enumerate(loaders[dsettype]):
                batch = batch.to(device_gpu)

                # 确保节点特征是long类型（用于embedding）
                batch.x_a = batch.x_a.long()
                batch.x_b = batch.x_b.long()
                with torch.no_grad():
                    if task == 'regression':
                        batch_pred_scores = hypergraph_model(batch)
                    elif use_reconstruction:
                        logsoftmax_scores, _, _ = hypergraph_model(batch)
                    else:
                        logsoftmax_scores = hypergraph_model(batch)

                if task == 'regression':
                    target_scores = get_regression_target(batch, target_score)
                    batch_pred_scores = batch_pred_scores * target_std_gpu + target_mean_gpu
                    pred_scores.extend(batch_pred_scores.detach().cpu().view(-1).tolist())
                    ref_scores.extend(target_scores.detach().cpu().view(-1).tolist())
                else:
                    __, y_pred_clss = torch.max(logsoftmax_scores, -1)
                    y_pred_prob = torch.exp(logsoftmax_scores.detach().cpu()).numpy()

                    pred_class.extend(y_pred_clss.view(-1).tolist())
                    ref_class.extend(batch.y.view(-1).tolist())
                    prob_scores.append(y_pred_prob)
                l_ids.extend(batch.id.view(-1).tolist())

            if task == 'regression':
                dset_perf = regression_metric_report(
                    pred_scores, ref_scores, epoch,
                    outlog=os.path.join(exp_dir, dsettype + ".log")
                )
            else:
                prob_scores_arr = np.concatenate(prob_scores, axis=0)
                dset_perf = perfmetric_report(pred_class, ref_class, prob_scores_arr[:,1], epoch,
                                              outlog=os.path.join(exp_dir, dsettype + ".log"))

            perfs[dsettype] = dset_perf

            if (dsettype == "valid"):
                if task == 'regression':
                    pass
                else:
                    fscore = F_score(perfs['valid'].s_aupr, perfs['valid'].s_auc)
                    if (fscore > best_fscore):
                        best_fscore = fscore
                        best_epoch = epoch

            if (dsettype == "test"):
                if task == 'regression':
                    predictions_df = build_regression_predictions_df(l_ids, ref_scores, pred_scores)
                else:
                    predictions_df = build_predictions_df(l_ids, ref_class, pred_class, prob_scores_arr)
                predictions_df.to_csv(os.path.join(exp_dir, 'predictions',
                                                   f'epoch_{epoch}_predictions_{dsettype}.csv'))

        print({'Train': perfs['train'], 'Validation': perfs['valid'], 'Test': perfs['test']})

        if task == 'regression':
            train_curve_rmse.append(perfs['train'].rmse)
            valid_curve_rmse.append(perfs['valid'].rmse)
            test_curve_rmse.append(perfs['test'].rmse)
            train_curve_mae.append(perfs['train'].mae)
            valid_curve_mae.append(perfs['valid'].mae)
            test_curve_mae.append(perfs['test'].mae)
            train_curve_r2.append(perfs['train'].r2)
            valid_curve_r2.append(perfs['valid'].r2)
            test_curve_r2.append(perfs['test'].r2)

            if perfs['valid'].rmse < best_valid_rmse:
                best_valid_rmse = perfs['valid'].rmse
                best_epoch = epoch
                epochs_without_improvement = 0
                save_best_checkpoint(epoch, 'valid_rmse', best_valid_rmse)
            else:
                epochs_without_improvement += 1
        else:
            train_curve_aupr.append(perfs['train'].s_aupr)
            valid_curve_aupr.append(perfs['valid'].s_aupr)
            test_curve_aupr.append(perfs['test'].s_aupr)

            train_curve_auc.append(perfs['train'].s_auc)
            valid_curve_auc.append(perfs['valid'].s_auc)
            test_curve_auc.append(perfs['test'].s_auc)

            if perfs['valid'].s_aupr > best_valid_aupr:
                best_valid_aupr = perfs['valid'].s_aupr
                epochs_without_improvement = 0
                save_best_checkpoint(epoch, 'valid_aupr', best_valid_aupr)
            else:
                epochs_without_improvement += 1

        if task == 'classification':
            early_stop_metric_desc = f"Best valid AUPR: {best_valid_aupr:.4f}"
        else:
            early_stop_metric_desc = f"Best valid RMSE: {best_valid_rmse:.4f}"

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}: validation metric did not improve for "
                f"{early_stopping_patience} epochs. {early_stop_metric_desc}"
            )
            break

    print('Finished training!')
    if task == 'regression':
        print(f'Best validation epoch: {best_epoch}, Best validation RMSE: {best_valid_rmse:.4f}')
    else:
        print(f'Best validation epoch: {best_epoch}, Best validation F-score: {best_fscore:.4f}')

    # 保存训练曲线
    if task == 'regression':
        df_curves = pd.DataFrame(np.array([
            train_curve_rmse, valid_curve_rmse, test_curve_rmse,
            train_curve_mae, valid_curve_mae, test_curve_mae,
            train_curve_r2, valid_curve_r2, test_curve_r2
        ]).T)
        df_curves.columns = [
            'train_rmse', 'valid_rmse', 'test_rmse',
            'train_mae', 'valid_mae', 'test_mae',
            'train_r2', 'valid_r2', 'test_r2'
        ]
    else:
        df_curves = pd.DataFrame(np.array([train_curve_aupr, valid_curve_aupr, test_curve_aupr,
                                           train_curve_auc, valid_curve_auc, test_curve_auc]).T)
        df_curves.columns = ['train_aupr', 'valid_aupr', 'test_aupr', 'train_auc', 'valid_auc', 'test_auc']
    df_curves.index.name = "epoch"
    df_curves.to_csv(exp_dir + "/curves.csv")
    sns.lineplot(data=df_curves).figure.savefig(exp_dir + "/curves.png")

    queue.put(gpu_num)
