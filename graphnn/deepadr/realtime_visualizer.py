"""
实时可视化训练进度
每个epoch后更新进度图
"""
import matplotlib.pyplot as plt
import pandas as pd
import os


class RealtimeVisualizer:
    """实时可视化训练进度"""

    def __init__(self, save_dir, exp_name):
        """
        Args:
            save_dir: 保存目录
            exp_name: 实验名称
        """
        self.save_dir = save_dir
        self.exp_name = exp_name
        self.curves_path = os.path.join(save_dir, 'curves.csv')
        self.plot_path = os.path.join(save_dir, 'training_progress.png')

        # 初始化curves.csv
        if not os.path.exists(self.curves_path):
            df = pd.DataFrame(columns=[
                'epoch', 'train_loss', 'train_aupr', 'train_auc',
                'valid_loss', 'valid_aupr', 'valid_auc',
                'test_loss', 'test_aupr', 'test_auc'
            ])
            df.to_csv(self.curves_path, index=False)

    def update(self, epoch, metrics):
        """
        更新一个epoch的数据并重新绘图

        Args:
            epoch: 当前epoch
            metrics: dict，包含所有指标
                {
                    'train_loss': float,
                    'train_aupr': float,
                    'train_auc': float,
                    'valid_loss': float,
                    'valid_aupr': float,
                    'valid_auc': float,
                    'test_loss': float,
                    'test_aupr': float,
                    'test_auc': float
                }
        """
        # 读取现有数据
        df = pd.read_csv(self.curves_path)

        # 添加新数据
        new_row = {'epoch': epoch}
        new_row.update(metrics)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # 保存更新后的数据
        df.to_csv(self.curves_path, index=False)

        # 重新绘图
        self._plot(df)

    def _plot(self, df):
        """绘制训练进度图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.exp_name} - Training Progress', fontsize=16)

        # 1. Loss曲线
        ax = axes[0, 0]
        ax.plot(df['epoch'], df['train_loss'], label='Train', marker='o', markersize=3)
        ax.plot(df['epoch'], df['valid_loss'], label='Valid', marker='s', markersize=3)
        ax.plot(df['epoch'], df['test_loss'], label='Test', marker='^', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. AUPR曲线
        ax = axes[0, 1]
        ax.plot(df['epoch'], df['train_aupr'], label='Train', marker='o', markersize=3)
        ax.plot(df['epoch'], df['valid_aupr'], label='Valid', marker='s', markersize=3)
        ax.plot(df['epoch'], df['test_aupr'], label='Test', marker='^', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUPR')
        ax.set_title('AUPR Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. AUC曲线
        ax = axes[1, 0]
        ax.plot(df['epoch'], df['train_auc'], label='Train', marker='o', markersize=3)
        ax.plot(df['epoch'], df['valid_auc'], label='Valid', marker='s', markersize=3)
        ax.plot(df['epoch'], df['test_auc'], label='Test', marker='^', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.set_title('AUC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. 最佳指标摘要
        ax = axes[1, 1]
        ax.axis('off')

        # 找到最佳epoch
        best_valid_aupr_idx = df['valid_aupr'].idxmax()
        best_epoch = df.loc[best_valid_aupr_idx, 'epoch']

        summary_text = f"""
Best Epoch: {int(best_epoch)}

Valid AUPR: {df.loc[best_valid_aupr_idx, 'valid_aupr']:.4f}
Valid AUC:  {df.loc[best_valid_aupr_idx, 'valid_auc']:.4f}

Test AUPR:  {df.loc[best_valid_aupr_idx, 'test_aupr']:.4f}
Test AUC:   {df.loc[best_valid_aupr_idx, 'test_auc']:.4f}

Current Epoch: {int(df['epoch'].iloc[-1])}
        """

        ax.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        plt.savefig(self.plot_path, dpi=100, bbox_inches='tight')
        plt.close()
