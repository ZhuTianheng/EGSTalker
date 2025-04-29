"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import torch
from torch import nn

# 自定义的 LayerNorm（层归一化）类
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        """
        初始化 LayerNorm 层

        Parameters:
        - d_model: 输入特征的维度，即最后一个维度的大小
        - eps: 一个小的常数，用于避免在计算标准差时出现除零错误
        """
        super(LayerNorm, self).__init__()
        # 可学习的缩放因子（gamma），初始化为全1
        self.gamma = nn.Parameter(torch.ones(d_model))
        # 可学习的偏移量（beta），初始化为全0
        self.beta = nn.Parameter(torch.zeros(d_model))
        # 防止除零错误的小常数
        self.eps = eps

    def forward(self, x):
        """
        前向传播，执行层归一化操作

        Parameters:
        - x: 输入张量，通常为形状 (batch_size, ..., d_model) 的张量

        Returns:
        - out: 经过归一化和缩放平移后的输出张量
        """
        # 计算输入的均值，-1表示对最后一个维度进行求均值，keepdim=True保留维度
        mean = x.mean(-1, keepdim=True)
        # 计算输入的方差，unbiased=False使用总体方差，keepdim=True保留维度
        var = x.var(-1, unbiased=False, keepdim=True)

        # 对输入进行归一化处理，标准化：x减去均值后除以标准差
        out = (x - mean) / torch.sqrt(var + self.eps)
        
        # 使用可学习的缩放因子（gamma）和平移量（beta）对归一化后的结果进行调整
        out = self.gamma * out + self.beta
        
        # 返回归一化并调整后的输出
        return out
