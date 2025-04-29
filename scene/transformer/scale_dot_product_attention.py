"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""

"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""

import math
from torch import nn

class ScaleDotProductAttention(nn.Module):
    """
    计算缩放点积注意力

    Query: 给定的关注句子（解码器）
    Key: 用于检查与查询的关系的每个句子（编码器）
    Value: 与键相同的每个句子（编码器）
    """

    def __init__(self):
        """
        初始化 ScaleDotProductAttention 层
        """
        super(ScaleDotProductAttention, self).__init__()
        # Softmax 层用于计算注意力权重
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        """
        前向传播，计算缩放点积注意力
        
        Parameters:
        - q: 查询张量，形状为 [batch_size, head, length, d_tensor]
        - k: 键张量，形状为 [batch_size, head, length, d_tensor]
        - v: 值张量，形状为 [batch_size, head, length, d_tensor]
        - mask: 可选的遮罩，形状为 [batch_size, head, length, length]
        - e: 一个小的常数，用于防止数值不稳定，默认值为 1e-12
        
        Returns:
        - v: 经过注意力加权后的值，形状为 [batch_size, head, length, d_tensor]
        - score: 注意力权重，形状为 [batch_size, head, length, length]
        """
        # 获取键张量的形状
        batch_size, head, length, d_tensor = k.size()

        # 1. 查询与键的转置进行点积以计算相似度
        k_t = k.transpose(2, 3)  # 将键张量在第 2 和第 3 维进行转置
        score = (q @ k_t) / math.sqrt(d_tensor)  # 缩放点积，q @ k_t 是点积操作，结果除以 sqrt(d_tensor)

        # 2. 应用遮罩（如果有的话）
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)  # 对于 mask 为 0 的位置，给出一个非常小的值（-10000）

        # 3. 对分数应用 softmax，得到 [0, 1] 范围的注意力权重
        score = self.softmax(score)

        # 4. 将注意力权重与值（v）相乘，得到加权后的值
        v = score @ v  # v 是按注意力权重加权的结果

        # 返回加权后的值和注意力权重
        #print(f"sdp计算后v和score的形状: {v.shape}，{score.shape}")  # v和score的形状
        return v, score

