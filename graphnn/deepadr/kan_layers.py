"""
KAN (Kolmogorov-Arnold Networks) Layers
支持B样条和傅里叶两种基函数实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class KANLinear(nn.Module):
    """
    KAN线性层的基类
    用可学习的激活函数替代传统的 Linear + Activation
    """
    def __init__(self, in_features, out_features, kan_type='fourier'):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kan_type = kan_type

        if kan_type == 'fourier':
            self.kan = FourierKAN(in_features, out_features)
        elif kan_type == 'bspline':
            self.kan = BSplineKAN(in_features, out_features)
        else:
            raise ValueError(f"Unknown KAN type: {kan_type}")

    def forward(self, x):
        return self.kan(x)


class FourierKAN(nn.Module):
    """
    基于傅里叶级数的KAN实现

    原理：将每个输入特征映射为傅里叶级数的线性组合
    f(x) = a0 + Σ(an*cos(n*ω*x) + bn*sin(n*ω*x))

    这种方法能够学习任意周期性和非周期性的复杂函数
    """
    def __init__(self, in_features, out_features, num_frequencies=5):  # 使用5个频率
        super(FourierKAN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_frequencies = num_frequencies

        # 为每个输入-输出对学习傅里叶系数
        # 形状: (out_features, in_features, 2*num_frequencies+1)
        # 2*num_frequencies+1 = a0 + (an, bn) * num_frequencies
        self.fourier_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, 2 * num_frequencies + 1) * 0.1
        )

        # 基础频率
        self.register_buffer('frequencies',
                           torch.arange(1, num_frequencies + 1).float())

        # 修复4: 添加LayerNorm，防止输出数值爆炸
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_features)
        Returns:
            out: (batch_size, out_features)
        """
        batch_size = x.shape[0]

        # 扩展维度以便计算
        # x: (batch, in_features) -> (batch, 1, in_features, 1)
        x_expanded = x.unsqueeze(1).unsqueeze(-1)

        # 计算傅里叶基函数
        # (batch, 1, in_features, 1) * (num_frequencies,) -> (batch, 1, in_features, num_frequencies)
        angles = x_expanded * self.frequencies * math.pi

        # 构建傅里叶基: [1, cos(ω*x), sin(ω*x), cos(2ω*x), sin(2ω*x), ...]
        # (batch, 1, in_features, 2*num_frequencies+1)
        fourier_basis = torch.cat([
            torch.ones_like(x_expanded[..., :1]),  # a0项
            torch.cos(angles),  # cos项
            torch.sin(angles)   # sin项
        ], dim=-1)

        # 与系数相乘并求和
        # fourier_basis: (batch, 1, in_features, 2*num_frequencies+1)
        # fourier_coeffs: (out_features, in_features, 2*num_frequencies+1)
        # 结果: (batch, out_features)
        out = torch.einsum('boif,oif->bo', fourier_basis, self.fourier_coeffs)

        # 修复4: LayerNorm归一化输出，防止数值爆炸
        out = self.layer_norm(out)

        return out


class BSplineKAN(nn.Module):
    """
    基于B样条的KAN实现

    原理：使用B样条基函数的线性组合来逼近任意函数
    B样条具有局部支撑性，计算效率高，且数值稳定
    """
    def __init__(self, in_features, out_features, num_knots=10, degree=3):
        super(BSplineKAN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots
        self.degree = degree

        # B样条的控制点数量
        num_control_points = num_knots + degree - 1

        # 为每个输入-输出对学习B样条系数
        # 形状: (out_features, in_features, num_control_points)
        self.spline_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, num_control_points) * 0.1
        )

        # 定义节点向量（knot vector）
        # 使用均匀节点分布在[-1, 1]区间
        knots = torch.linspace(-1, 1, num_knots)
        # 扩展节点向量以满足B样条边界条件
        knots = torch.cat([
            knots[0].repeat(degree),
            knots,
            knots[-1].repeat(degree)
        ])
        self.register_buffer('knots', knots)

    def b_spline_basis(self, x, i, k):
        """
        递归计算B样条基函数

        Args:
            x: 输入值
            i: 基函数索引
            k: B样条阶数（degree）
        """
        if k == 0:
            # 0阶B样条（分段常数）
            return ((x >= self.knots[i]) & (x < self.knots[i + 1])).float()
        else:
            # 递归计算高阶B样条
            # Cox-de Boor递归公式
            denom1 = self.knots[i + k] - self.knots[i]
            denom2 = self.knots[i + k + 1] - self.knots[i + 1]

            term1 = 0
            if denom1 > 1e-6:
                term1 = ((x - self.knots[i]) / denom1) * self.b_spline_basis(x, i, k - 1)

            term2 = 0
            if denom2 > 1e-6:
                term2 = ((self.knots[i + k + 1] - x) / denom2) * self.b_spline_basis(x, i + 1, k - 1)

            return term1 + term2

    def forward(self, x):
        """
        Args:
            x: (batch_size, in_features)
        Returns:
            out: (batch_size, out_features)
        """
        batch_size = x.shape[0]

        # 将输入归一化到[-1, 1]区间
        x_normalized = torch.tanh(x)

        # 计算所有B样条基函数的值
        # (batch, in_features, num_control_points)
        num_control_points = self.spline_coeffs.shape[2]
        basis_values = []

        for i in range(num_control_points):
            # (batch, in_features)
            basis = self.b_spline_basis(x_normalized, i, self.degree)
            basis_values.append(basis)

        # (batch, in_features, num_control_points)
        basis_matrix = torch.stack(basis_values, dim=-1)

        # 与系数相乘并求和
        # basis_matrix: (batch, in_features, num_control_points)
        # spline_coeffs: (out_features, in_features, num_control_points)
        # 结果: (batch, out_features)
        out = torch.einsum('bic,oic->bo', basis_matrix, self.spline_coeffs)

        return out
