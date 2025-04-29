"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

"""
from torch import nn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
"""

"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

from torch import nn

# 定义 PositionwiseFeedForward 类，继承自 nn.Module
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        """
        初始化 Positionwise Feed Forward 层
        
        Parameters:
        - d_model: 输入的特征维度
        - hidden: 隐藏层的特征维度，通常是 d_model 的倍数
        - drop_prob: Dropout 的概率，默认为 0.1
        """
        super(PositionwiseFeedForward, self).__init__()
        
        # 第一个全连接层，将输入特征维度映射到隐藏层维度
        self.linear1 = nn.Linear(d_model, hidden)
        
        # 第二个全连接层，将隐藏层维度映射回输入特征维度
        self.linear2 = nn.Linear(hidden, d_model)
        
        # ReLU 激活函数
        self.relu = nn.ReLU()
        
        # Dropout 层，用于防止过拟合
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        前向传播，执行 Positionwise Feed Forward 操作
        
        Parameters:
        - x: 输入张量，形状为 [batch_size, seq_len, d_model]
        
        Returns:
        - x: 经过位置前馈神经网络处理后的输出，形状为 [batch_size, seq_len, d_model]
        """
        # 1. 输入 x 经过第一个全连接层
        x = self.linear1(x)
        
        # 2. 通过 ReLU 激活函数进行非线性变换
        x = self.relu(x)
        
        # 3. 进行 Dropout 操作，防止过拟合
        x = self.dropout(x)
        
        # 4. 输入经过第二个全连接层，映射回原始的维度 d_model
        x = self.linear2(x)
        
        # 5. 返回处理后的输出
        return x

