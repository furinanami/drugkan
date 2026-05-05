"""
Cold场景数据分区
支持 random / cold drug / cold cell 三种场景
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_cold_drug_split(dataset, fold=0, n_folds=5, seed=42):
    """
    Cold Drug场景：测试集中的药物在训练集中完全没见过

    Args:
        dataset: MoleculeDataset对象
        fold: 当前折数
        n_folds: 总折数
        seed: 随机种子

    Returns:
        partition: {'train': [...], 'valid': [...], 'test': [...]}
    """
    np.random.seed(seed + fold)

    # 获取所有唯一的药物ID
    drug_ids_a = dataset.data.drug_a_id.cpu().numpy()
    drug_ids_b = dataset.data.drug_b_id.cpu().numpy()
    all_drug_ids = np.unique(np.concatenate([drug_ids_a, drug_ids_b]))

    # 随机打乱药物ID
    np.random.shuffle(all_drug_ids)

    # 分割药物ID
    n_drugs = len(all_drug_ids)
    test_size = n_drugs // n_folds
    test_start = fold * test_size
    test_end = test_start + test_size if fold < n_folds - 1 else n_drugs

    test_drugs = set(all_drug_ids[test_start:test_end])
    train_valid_drugs = set(all_drug_ids) - test_drugs

    # 从train_valid中分出valid（10%）
    train_valid_drugs_list = list(train_valid_drugs)
    np.random.shuffle(train_valid_drugs_list)
    valid_size = len(train_valid_drugs_list) // 10
    valid_drugs = set(train_valid_drugs_list[:valid_size])
    train_drugs = train_valid_drugs - valid_drugs

    # 分配样本
    train_indices = []
    valid_indices = []
    test_indices = []

    for i in range(len(dataset)):
        drug_a = int(drug_ids_a[i])
        drug_b = int(drug_ids_b[i])

        # 如果任一药物在测试集中，该样本属于测试集
        if drug_a in test_drugs or drug_b in test_drugs:
            test_indices.append(i)
        # 如果任一药物在验证集中（且不在测试集），该样本属于验证集
        elif drug_a in valid_drugs or drug_b in valid_drugs:
            valid_indices.append(i)
        # 否则属于训练集
        else:
            train_indices.append(i)

    return {
        'train': train_indices,
        'valid': valid_indices,
        'test': test_indices
    }


def get_cold_cell_split(dataset, fold=0, n_folds=5, seed=42):
    """
    Cold Cell场景：测试集中的细胞系在训练集中完全没见过

    Args:
        dataset: MoleculeDataset对象
        fold: 当前折数
        n_folds: 总折数
        seed: 随机种子

    Returns:
        partition: {'train': [...], 'valid': [...], 'test': [...]}
    """
    np.random.seed(seed + fold)

    # 获取所有唯一的细胞系ID
    cell_ids = dataset.data.cell_id.cpu().numpy()
    unique_cells = np.unique(cell_ids)

    # 随机打乱细胞系ID
    np.random.shuffle(unique_cells)

    # 分割细胞系ID
    n_cells = len(unique_cells)
    test_size = n_cells // n_folds
    test_start = fold * test_size
    test_end = test_start + test_size if fold < n_folds - 1 else n_cells

    test_cells = set(unique_cells[test_start:test_end])
    train_valid_cells = set(unique_cells) - test_cells

    # 从train_valid中分出valid（10%）
    train_valid_cells_list = list(train_valid_cells)
    np.random.shuffle(train_valid_cells_list)
    valid_size = len(train_valid_cells_list) // 10
    valid_cells = set(train_valid_cells_list[:valid_size])
    train_cells = train_valid_cells - valid_cells

    # 分配样本
    train_indices = []
    valid_indices = []
    test_indices = []

    for i in range(len(dataset)):
        cell = int(cell_ids[i])

        if cell in test_cells:
            test_indices.append(i)
        elif cell in valid_cells:
            valid_indices.append(i)
        else:
            train_indices.append(i)

    return {
        'train': train_indices,
        'valid': valid_indices,
        'test': test_indices
    }


def get_random_split(dataset, fold=0, n_folds=5, seed=42):
    """
    Random场景：随机分割（原有的stratified split）

    Args:
        dataset: MoleculeDataset对象
        fold: 当前折数
        n_folds: 总折数
        seed: 随机种子

    Returns:
        partition: {'train': [...], 'valid': [...], 'test': [...]}
    """
    from sklearn.model_selection import StratifiedKFold
    import torch

    # 获取标签
    y = dataset.data.y.cpu().numpy()
    indices = np.arange(len(dataset))

    # 创建分层K折
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # 获取当前折的训练+验证集和测试集
    for i, (train_valid_idx, test_idx) in enumerate(skf.split(indices, y)):
        if i == fold:
            # 从训练+验证集中再分出验证集（10%）
            train_valid_y = y[train_valid_idx]

            # 再次分层分割
            skf_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
            for j, (train_idx_inner, valid_idx_inner) in enumerate(skf_inner.split(train_valid_idx, train_valid_y)):
                if j == 0:  # 只取第一折作为验证集
                    train_idx = train_valid_idx[train_idx_inner]
                    valid_idx = train_valid_idx[valid_idx_inner]
                    break

            return {
                'train': train_idx.tolist(),
                'valid': valid_idx.tolist(),
                'test': test_idx.tolist()
            }

    # 如果没有找到对应的fold，返回空
    raise ValueError(f"Fold {fold} not found in {n_folds} folds")


def get_split_by_scenario(dataset, scenario='random', fold=0, n_folds=5, seed=42):
    """
    根据场景选择分割方法

    Args:
        dataset: MoleculeDataset对象
        scenario: 'random' / 'cold_drug' / 'cold_cell'
        fold: 当前折数
        n_folds: 总折数
        seed: 随机种子

    Returns:
        partition: {'train': [...], 'valid': [...], 'test': [...]}
    """
    if scenario == 'random':
        return get_random_split(dataset, fold, n_folds, seed)
    elif scenario == 'cold_drug':
        return get_cold_drug_split(dataset, fold, n_folds, seed)
    elif scenario == 'cold_cell':
        return get_cold_cell_split(dataset, fold, n_folds, seed)
    else:
        raise ValueError(f"Unknown scenario: {scenario}. Must be 'random', 'cold_drug', or 'cold_cell'.")
