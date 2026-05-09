"""
Cold场景数据分区
支持 random / cold drug / cold cell 三种场景
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold


def _as_int_set(values):
    return set(int(x) for x in values)


def _build_drug_index(drug_ids_a, drug_ids_b, indices):
    drug_to_indices = {}
    for idx in indices:
        idx = int(idx)
        drug_a = int(drug_ids_a[idx])
        drug_b = int(drug_ids_b[idx])
        drug_to_indices.setdefault(drug_a, set()).add(idx)
        drug_to_indices.setdefault(drug_b, set()).add(idx)
    return drug_to_indices


def _score_exposure_split(count, pos_count, target_count, target_pos_rate):
    if target_count <= 0:
        return float(count)
    count_error = abs(count - target_count) / float(target_count)
    if count > target_count:
        count_error *= 1.05
    if count == 0:
        pos_error = 1.0
    else:
        pos_error = abs((pos_count / float(count)) - target_pos_rate)
    return count_error + 0.25 * pos_error


def _select_drugs_for_target(drug_to_indices, candidate_drugs, y, target_count, rng, target_pos_rate):
    """Greedily choose drugs whose exposure samples are close to target_count."""
    candidate_drugs = [int(d) for d in candidate_drugs if int(d) in drug_to_indices]
    rng.shuffle(candidate_drugs)

    selected_drugs = set()
    selected_indices = set()
    selected_pos = 0.0
    current_score = _score_exposure_split(0, 0.0, target_count, target_pos_rate)

    while candidate_drugs:
        best = None
        for drug in candidate_drugs:
            if drug in selected_drugs:
                continue
            marginal = drug_to_indices[drug] - selected_indices
            if not marginal:
                continue
            marginal_list = list(marginal)
            new_count = len(selected_indices) + len(marginal_list)
            new_pos = selected_pos + float(y[marginal_list].sum())
            score = _score_exposure_split(new_count, new_pos, target_count, target_pos_rate)
            count_error = abs(new_count - target_count)
            tie_break = (score, count_error, len(marginal_list), drug)
            if best is None or tie_break < best[0]:
                best = (tie_break, drug, marginal, new_pos)

        if best is None:
            break

        best_score, _, _, _ = best[0]
        if selected_indices and best_score >= current_score:
            break

        _, drug, marginal, new_pos = best
        selected_drugs.add(drug)
        selected_indices.update(marginal)
        selected_pos = new_pos
        current_score = best_score

        if len(selected_indices) >= target_count and best_score < 0.01:
            break

    return selected_drugs, selected_indices


def get_cold_drug_split_strict(dataset, fold=0, n_folds=5, seed=42):
    """
    Strict Cold Drug场景：测试集中的所有药物在训练集中完全没见过。

    严格划分规则：
    - train 样本的两个药物都必须来自 train_drugs。
    - valid 样本的两个药物都必须来自 valid_drugs。
    - test 样本的两个药物都必须来自 test_drugs。
    - 跨集合的混合药物对会被丢弃，避免 train-test 药物泄漏。

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

        if drug_a in test_drugs and drug_b in test_drugs:
            test_indices.append(i)
        elif drug_a in valid_drugs and drug_b in valid_drugs:
            valid_indices.append(i)
        elif drug_a in train_drugs and drug_b in train_drugs:
            train_indices.append(i)

    return {
        'train': train_indices,
        'valid': valid_indices,
        'test': test_indices
    }


def get_cold_drug_split(dataset, fold=0, n_folds=5, seed=42):
    """
    Balanced Cold Drug场景：测试/验证样本包含指定 held-out drug，且这些
    held-out drug 不会出现在训练集中；样本量按 random split 的
    train/valid/test 数量配平。

    与 strict cold-drug 不同，这里保留“held-out drug + seen partner”的样本，
    因此不会丢弃大量跨药物集合样本，训练/验证/测试规模与 random 更接近。
    """
    rng = np.random.RandomState(seed + fold)

    drug_ids_a = dataset.data.drug_a_id.cpu().numpy()
    drug_ids_b = dataset.data.drug_b_id.cpu().numpy()
    y = dataset.data.y.cpu().numpy().reshape(-1)
    all_indices = set(range(len(dataset)))
    all_drug_ids = _as_int_set(np.unique(np.concatenate([drug_ids_a, drug_ids_b])))

    reference = get_random_split(dataset, fold=fold, n_folds=n_folds, seed=seed)
    target_train = len(reference['train'])
    target_valid = len(reference['valid'])
    target_test = len(reference['test'])
    target_pos_rate = float(y.mean()) if len(y) else 0.0

    drug_to_indices = _build_drug_index(drug_ids_a, drug_ids_b, all_indices)
    test_drugs, test_indices = _select_drugs_for_target(
        drug_to_indices=drug_to_indices,
        candidate_drugs=list(all_drug_ids),
        y=y,
        target_count=target_test,
        rng=rng,
        target_pos_rate=target_pos_rate,
    )

    remaining_after_test = all_indices - test_indices
    valid_candidate_drugs = all_drug_ids - test_drugs
    valid_drug_to_indices = _build_drug_index(drug_ids_a, drug_ids_b, remaining_after_test)
    valid_drugs, valid_indices = _select_drugs_for_target(
        drug_to_indices=valid_drug_to_indices,
        candidate_drugs=list(valid_candidate_drugs),
        y=y,
        target_count=target_valid,
        rng=rng,
        target_pos_rate=target_pos_rate,
    )

    train_indices = all_indices - test_indices - valid_indices

    return {
        'train': sorted(int(i) for i in train_indices),
        'valid': sorted(int(i) for i in valid_indices),
        'test': sorted(int(i) for i in test_indices),
        'split_strategy': 'cold_drug_balanced_exposure',
        'target_sizes': {
            'train': target_train,
            'valid': target_valid,
            'test': target_test,
        },
        'heldout_test_drugs': sorted(int(d) for d in test_drugs),
        'heldout_validation_drugs': sorted(int(d) for d in valid_drugs),
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
        scenario: 'random' / 'cold_drug' / 'cold_drug_strict' / 'cold_cell'
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
    elif scenario == 'cold_drug_strict':
        return get_cold_drug_split_strict(dataset, fold, n_folds, seed)
    elif scenario == 'cold_cell':
        return get_cold_cell_split(dataset, fold, n_folds, seed)
    else:
        raise ValueError(
            f"Unknown scenario: {scenario}. Must be 'random', 'cold_drug', "
            "'cold_drug_strict', or 'cold_cell'."
        )
