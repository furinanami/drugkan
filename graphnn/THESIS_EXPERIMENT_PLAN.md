# 本科毕业论文实验计划：KAN/超图药物协同预测

目标：证明你的方法不只是“换了一个模型”，而是在药物协同预测任务上同时具备预测性能、泛化能力、模块必要性和可解释性。

## 1. 主结果实验

必须做。论文主表应该放这里。

### 数据划分

- Random split：常规性能。
- Cold-drug split：新药泛化，最重要。
- Cold-cell split：新细胞系泛化，如果时间允许做。

### 任务

- 分类：协同/非协同，指标 AUPR、AUC、F1、Precision、Recall。
- 回归：Loewe score，指标 RMSE、MAE、R2、Pearson、Spearman。

### 推荐主表

| Model | Split | Task | AUPR/AUC or RMSE/MAE | Notes |
|---|---|---|---|---|
| DeepSynergy/MLP | random/cold-drug | cls/reg | | 无图 baseline |
| GCN + MLP | random/cold-drug | cls/reg | | 普通分子图 |
| KAGCN + MLP | random/cold-drug | cls/reg | | 只加 drug KAN |
| GCN + Hypergraph | random/cold-drug | cls/reg | | 只加三元图结构 |
| KAGCN + KAN-Hypergraph | random/cold-drug | cls/reg | | 完整模型 |

结论要回答：完整模型是否在 random 和 cold-drug 上都稳定优于 baseline。

## 2. 模块消融实验

必须做。优秀论文需要证明每个模块有用。

### KAN 位置消融

| Drug Encoder KAN | Hypergraph KAN | Expected Question |
|---|---|---|
| off | off | 纯 baseline |
| on | off | KAN 是否提升药物结构表征 |
| off | on | KAN 是否提升 drug-drug-cell 交互建模 |
| on | on | 两者是否互补 |

### 超图结构消融

- `mlp`：无图融合。
- `hypergraph`：普通 drug A-drug B-cell 超边。
- `kan_hypergraph`：超图消息函数用 KAN。
- `kan_aggregated`：drug pair-cell 聚合图。

结论要回答：提升来自 KAN、超图结构，还是二者共同作用。

## 3. KAN 设计消融

建议做，能突出你自己的模型设计。

### KAN 类型

- Fourier KAN。
- B-spline KAN。

比较：

- 性能。
- 训练稳定性。
- 参数量/训练时间。
- 解释曲线复杂度。

### GNN 类型

- GCN。
- KAGCN。
- KAGCN-neighbor。
- GIN/KAGIN，如果已有结果可补。

重点不是所有都赢，而是解释为什么最终选某个结构。

## 4. 泛化与鲁棒性实验

优秀论文强烈建议做。

### 多随机种子

至少 3 个 seed，最好 5 个 seed。报告 mean ± std。

必须覆盖：

- 最强 baseline。
- 完整模型。
- random split。
- cold-drug split。

### 数据泄漏检查

需要在论文里明确：

- cold-drug 中 test drugs 不出现在 train。
- drug pair 不重复泄漏。
- cell line split 时 test cells 不出现在 train。

这个可以作为方法可靠性的一个小节。

## 5. 可解释性实验

必须做，因为你用了 KAN。

### KAN 函数曲线

使用 `explain_kan_functions.py`：

- 展示 top KAN channel 的函数曲线。
- 展示 layer-level nonlinear strength。
- 展示 input-channel importance。

论文图建议：

- 1 张 KAN 曲线网格图。
- 1 张不同层 KAN nonlinear strength barplot。

### 单样本三方贡献

使用 `explain_synergy_sample.py`：

- mask drug A。
- mask drug B。
- mask cell。
- mask both drugs。

论文里挑 2-3 个样本：

- 高协同预测正确样本。
- 非协同预测正确样本。
- 模型错误样本，用于讨论局限。

### 原子级 saliency

输出 `atom_embedding_saliency.csv` 后，后续最好接 RDKit 高亮图：

- drug A 关键原子。
- drug B 关键原子。

### 基因/通路重要性

先做 gene-level gradient ranking。

如果时间允许，再把 top genes 聚合到 pathway：

- KEGG。
- Reactome。
- MSigDB Hallmark。

这会显著提升生物学解释质量。

## 6. 案例研究

建议做 2-4 个高质量 case study。

每个案例包含：

- 药物 A / 药物 B / cell line。
- 真实 label 或 Loewe score。
- 模型预测。
- 三方扰动结果。
- KAN 曲线或重要通道。
- 原子 saliency。
- gene/pathway importance。
- 简短生物学解释。

案例研究是本科论文从“实验报告”变成“研究论文”的关键。

## 7. 参数量与效率

建议做，放补充表或结果小节。

比较：

- 参数量。
- 单 epoch 时间。
- 推理时间。
- GPU 显存。

需要证明 KAN/超图带来的性能提升不是用巨大计算成本换来的。

## 8. 最低完成版本

如果时间紧，至少完成：

1. Random + cold-drug 分类主结果。
2. Random + cold-drug Loewe 回归主结果。
3. KAN 位置消融：drug KAN、hypergraph KAN、both。
4. `mlp / hypergraph / kan_hypergraph` 超图消融。
5. 3 seeds 的 mean ± std。
6. KAN 曲线解释。
7. 2 个 case study。

这套已经足够支撑一篇质量较高的本科论文。

## 8.1 推荐运行命令

先检查划分是否无泄漏：

```bash
python3 graphnn/check_split_integrity.py \
  --dataset-root graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1 \
  --seeds 42,43,44 \
  --output graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1/experiments/split_integrity_thesis.csv
```

生成最小论文实验命令：

```bash
python3 graphnn/run_thesis_experiments.py \
  --gpus 0,1,2,3,4,5 \
  --seeds 42,43,44 \
  --epochs 80 \
  --preset minimal
```

确认 `graphnn/thesis_experiment_commands.txt` 后，在 GPU 机器上执行：

```bash
python3 graphnn/run_thesis_experiments.py \
  --gpus 0,1,2,3,4,5 \
  --seeds 42,43,44 \
  --epochs 80 \
  --preset minimal \
  --execute
```

训练完成后汇总所有结果：

```bash
python3 graphnn/collect_thesis_results.py \
  --experiments-root graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1/experiments \
  --output-dir graphnn/data/processed/DrugComb_loewe_thresh_1/data_v1/experiments/thesis_summary
```

## 9. 最推荐论文故事线

1. 药物协同预测需要同时建模药物结构、细胞系背景和三元交互。
2. 普通 MLP/GCN 对复杂非线性交互表达不足。
3. KAN 用可学习函数增强药物图消息传递和超图消息传递。
4. 超图显式建模 drug A-drug B-cell 三元关系。
5. 实验显示完整模型在 random 和 cold-drug 上更好。
6. 消融证明 KAN 与超图都必要。
7. 解释性实验展示模型关注的 KAN 非线性函数、关键原子和关键表达特征。
