# 对比学习框架方案

## 1. 核心原理

### 1.1 什么是对比学习？

**核心思想**：通过对比正样本对和负样本对，学习有意义的表示空间。

**在药物协同预测中的应用**：
- **正样本对**：有协同效应的药物对 (Drug A, Drug B, synergy=1)
- **负样本对**：无协同效应的药物对 (Drug A, Drug B, synergy=0)
- **目标**：让有协同效应的药物对在表示空间中距离更近，无协同效应的距离更远

### 1.2 为什么对比学习有效？

**问题1：标签不平衡**
- 你的数据：class 0 (无协同) ~85%, class 1 (有协同) ~15%
- 传统交叉熵损失容易偏向多数类

**解决**：对比学习不直接依赖标签，而是学习样本间的相对关系

**问题2：特征表示不够判别性**
- GNN学到的drug embedding可能对协同/非协同不够敏感

**解决**：对比损失强制模型学习判别性特征

**问题3：缺乏大规模标注数据**
- 药物协同实验昂贵，标注数据有限

**解决**：自监督预训练 + 少量标注数据微调

### 1.3 对比学习的数学原理

#### InfoNCE Loss (最常用)

```python
给定一个anchor样本 x_i，一个正样本 x_i+，N个负样本 {x_j-}

相似度：sim(u, v) = u^T v / (||u|| ||v||)  # cosine similarity

InfoNCE Loss:
L = -log( exp(sim(z_i, z_i+) / τ) / 
          (exp(sim(z_i, z_i+) / τ) + Σ_j exp(sim(z_i, z_j-) / τ)) )

其中：
- z_i, z_i+, z_j- 是样本的embedding
- τ 是温度参数（控制分布的平滑度）
```

**直观理解**：
- 分子：正样本对的相似度要大
- 分母：正样本相似度 vs 所有负样本相似度
- 目标：最大化正样本相似度，同时最小化负样本相似度

#### Triplet Loss

```python
给定三元组：(anchor, positive, negative)

L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

其中 d(·,·) 是距离函数（如欧氏距离）
```

**直观理解**：
- anchor到positive的距离 < anchor到negative的距离 + margin
- margin是安全边界

#### Supervised Contrastive Loss

```python
利用标签信息的对比学习：
- 同一类别的所有样本都是正样本
- 不同类别的所有样本都是负样本

L = -Σ_{p∈P(i)} log( exp(sim(z_i, z_p) / τ) / 
                      Σ_{a∈A(i)} exp(sim(z_i, z_a) / τ) )

其中：
- P(i) = {p: y_p = y_i, p ≠ i}  # 同类样本
- A(i) = {a: a ≠ i}              # 所有其他样本
```

## 2. 在DDoS中的应用策略

### 策略1：药物对级别的对比学习

**构造样本对**：
```python
# 正样本对：同一药物对在不同细胞系中的表现
anchor: (Drug A, Drug B, Cell Line 1) → synergy
positive: (Drug A, Drug B, Cell Line 2) → synergy

# 负样本对：不同药物对
negative: (Drug C, Drug D, Cell Line X) → no synergy
```

**优势**：
- 学习药物对的通用协同模式
- 跨细胞系的泛化能力

### 策略2：药物级别的对比学习（自监督预训练）

**数据增强**：
```python
# 同一药物的不同"视图"
Drug A 原始图 → 数据增强 → Drug A'

增强方法：
1. 节点dropout：随机删除部分原子
2. 边dropout：随机删除部分化学键
3. 子图采样：提取不同的子结构
4. 特征mask：随机mask部分原子特征
```

**对比目标**：
```python
# 同一药物的不同增强视图是正样本对
positive: (Drug A view1, Drug A view2)

# 不同药物是负样本对
negative: (Drug A view1, Drug B view1)
```

**预训练流程**：
```
Step 1: 在大规模无标签药物图上预训练GNN encoder
        使用对比损失，学习通用的药物表示

Step 2: 在药物协同数据上微调
        固定或微调GNN encoder，训练下游分类器
```

### 策略3：多视图对比学习

**多个视图**：
- 视图1：分子图结构（GNN）
- 视图2：分子指纹（Morgan fingerprint）
- 视图3：药物知识图谱嵌入
- 视图4：药物-靶点相互作用

**对比目标**：
```python
# 同一药物的不同视图应该有相似的表示
L_cross_view = InfoNCE(view1, view2) + InfoNCE(view1, view3) + ...
```

## 3. 完整架构设计

### 3.1 预训练阶段（Self-Supervised）

```
输入：大规模药物分子图（无标签）

Pipeline:
1. 数据增强：生成两个增强视图
   Drug → Augment → Drug_view1, Drug_view2

2. GNN编码：
   z1 = GNN_encoder(Drug_view1)
   z2 = GNN_encoder(Drug_view2)

3. 投影头（Projection Head）：
   h1 = MLP_proj(z1)  # 映射到对比学习空间
   h2 = MLP_proj(z2)

4. 对比损失：
   L_contrastive = InfoNCE(h1, h2)

5. 优化：
   更新GNN_encoder和MLP_proj的参数
```

**关键组件**：
```python
class ContrastivePretraining(nn.Module):
    def __init__(self, gnn_encoder, proj_dim=128):
        self.encoder = gnn_encoder  # 共享的GNN
        self.projection_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, proj_dim)
        )
        
    def forward(self, drug_view1, drug_view2):
        # 编码两个视图
        z1 = self.encoder(drug_view1)
        z2 = self.encoder(drug_view2)
        
        # 投影到对比学习空间
        h1 = self.projection_head(z1)
        h2 = self.projection_head(z2)
        
        return h1, h2
```

### 3.2 微调阶段（Supervised）

```
输入：药物协同数据（有标签）

Pipeline:
1. 加载预训练的GNN encoder
2. 冻结或微调encoder
3. 添加下游任务头（分类器）
4. 使用交叉熵 + 监督对比损失训练

L_total = L_CE + λ * L_supervised_contrastive
```

**监督对比损失**：
```python
# 同一类别（synergy=1）的药物对互为正样本
# 不同类别的药物对互为负样本

for batch in dataloader:
    embeddings = model.get_embeddings(batch)
    labels = batch.y
    
    # 计算监督对比损失
    loss_contrast = supervised_contrastive_loss(
        embeddings, labels, temperature=0.07
    )
    
    # 计算分类损失
    logits = model.classifier(embeddings)
    loss_ce = cross_entropy(logits, labels)
    
    # 总损失
    loss = loss_ce + 0.5 * loss_contrast
```

## 4. 数据增强策略

### 4.1 图结构增强

```python
def augment_graph(data):
    """
    对分子图进行增强
    """
    # 1. 节点dropout（删除原子）
    if random.random() < 0.2:
        data = node_dropout(data, drop_ratio=0.1)
    
    # 2. 边dropout（删除化学键）
    if random.random() < 0.2:
        data = edge_dropout(data, drop_ratio=0.1)
    
    # 3. 子图采样
    if random.random() < 0.2:
        data = subgraph_sampling(data, ratio=0.8)
    
    # 4. 特征mask
    if random.random() < 0.2:
        data = feature_masking(data, mask_ratio=0.1)
    
    return data
```

### 4.2 增强的原则

**保持语义不变**：
- 删除少量原子/键不改变药物的主要性质
- 子结构仍然保留药效团

**多样性**：
- 不同的增强方法捕获不同的不变性
- 组合使用效果更好

## 5. 实施步骤

### Phase 1: 实现对比学习基础设施（3-4天）

1. **数据增强模块**
   - 实现图增强函数
   - 创建增强数据集类

2. **对比损失函数**
   - InfoNCE Loss
   - Supervised Contrastive Loss
   - Triplet Loss

3. **投影头**
   - MLP projection head
   - 归一化层

### Phase 2: 自监督预训练（2-3天）

1. **准备预训练数据**
   - 收集大规模药物分子（PubChem, ChEMBL）
   - 或使用现有的DrugComb中的所有药物

2. **预训练脚本**
   - 实现预训练循环
   - 保存checkpoint

3. **评估预训练质量**
   - 可视化学到的表示（t-SNE）
   - 线性探测（linear probing）

### Phase 3: 监督微调（2-3天）

1. **加载预训练模型**
2. **实现监督对比学习**
3. **联合训练（CE + Contrastive）**

### Phase 4: 实验对比（3-4天）

**对比实验**：
1. Baseline: 原始DDoS（无对比学习）
2. +Pretrain: 自监督预训练 + 微调
3. +SupCon: 监督对比学习（无预训练）
4. +Both: 预训练 + 监督对比学习
5. +Augmentation: 不同增强策略的消融

**评估指标**：
- AUROC, AUPRC, F1
- 少样本学习性能（10%, 20%, 50%数据）
- 跨细胞系泛化能力

## 6. 预期效果

### 性能提升
- **AUROC**: +3-7%（尤其在少样本场景）
- **AUPRC**: +5-10%（对不平衡数据特别有效）
- **泛化能力**: 跨细胞系预测提升显著

### 数据效率
- 使用50%数据达到原模型100%数据的性能
- 少样本学习能力增强

### 表示质量
- 学到的drug embedding更具判别性
- 相似药物在表示空间中聚类

### 创新点
- 首次在药物协同预测中应用对比学习
- 自监督预训练利用大规模无标签数据
- 多视图对比学习融合多模态信息

## 7. 进阶方向

### 7.1 Hard Negative Mining
- 挖掘难以区分的负样本
- 提升模型判别能力

### 7.2 Momentum Contrast (MoCo)
- 使用动量编码器
- 维护大规模负样本队列

### 7.3 对比学习 + 图生成
- 生成具有协同效应的药物对
- 药物发现应用
