"""
Early Stopping机制
"""
import numpy as np
import torch


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=10, min_delta=0.0001, mode='max', verbose=True):
        """
        Args:
            patience: 容忍多少个epoch没有改善
            min_delta: 最小改善幅度
            mode: 'max' (越大越好，如AUPR) 或 'min' (越小越好，如loss)
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

        if mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
        else:
            self.is_better = lambda new, best: new < best - min_delta

    def __call__(self, score, epoch):
        """
        检查是否应该早停

        Args:
            score: 当前指标值
            epoch: 当前epoch

        Returns:
            should_stop: 是否应该停止训练
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.is_better(score, self.best_score):
            # 有改善
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  [EarlyStopping] 指标改善到 {score:.4f}")
        else:
            # 没有改善
            self.counter += 1
            if self.verbose:
                print(f"  [EarlyStopping] 无改善 ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  [EarlyStopping] 触发早停！最佳epoch: {self.best_epoch}, 最佳分数: {self.best_score:.4f}")
                return True

        return False

    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
