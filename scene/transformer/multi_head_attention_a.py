"""
@author : Hyunwoong
@when : 2019-10-25
@homepage : https://github.com/gusdnd852
"""

from torch import nn
from scene.transformer.scale_dot_product_attention import ScaleDotProductAttention
from scene.transformer.agent_attention_m import AgentScaleDotProductAttention

# 定义 MultiHeadAttention 类，继承自 nn.Module
class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head, agent_num=400):
        """
        初始化 MultiHeadAttention 层

        Parameters:
        - d_model: 输入的特征维度
        - n_head: 多头注意力机制中的头数
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model  #agent参数
        self.agent_num = agent_num #agent参数
        
        # 创建一个 ScaleDotProductAttention 实例，用于计算缩放点积注意力
        #self.attention = ScaleDotProductAttention()
        self.attention = AgentScaleDotProductAttention(d_model=d_model // n_head, num_heads=n_head, agent_num=agent_num)
        
        # 定义线性变换层，用于对输入的查询（q）、键（k）和值（v）进行变换
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 连接头后的输出，并映射回原始的特征维度
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """
        前向传播，执行多头注意力计算

        Parameters:
        - q: 查询张量，形状为 [batch_size, seq_len, d_model]
        - k: 键张量，形状为 [batch_size, seq_len, d_model]
        - v: 值张量，形状为 [batch_size, seq_len, d_model]
        - mask: 可选的注意力遮罩，形状为 [batch_size, seq_len, seq_len]

        Returns:
        - out: 注意力输出，形状为 [batch_size, seq_len, d_model]
        - attention: 注意力权重矩阵，形状为 [batch_size, n_head, seq_len, seq_len]
        """
        # 1. 对查询（q）、键（k）和值（v）进行线性变换
        print(f"qkv线性变换前的形状: {q.shape}，{k.shape}，{v.shape}")  # 打印qkv线性变换后的形状
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        print(f"qkv线性变换后的形状: {q.shape}，{k.shape}，{v.shape}")  # 打印qkv线性变换后的形状


        # 2. 将查询（q）、键（k）和值（v）拆分成多个头
        q, k, v = self.split(q), self.split(k), self.split(v)
        #print(f"qkv拆分后的形状: {q.shape}，{k.shape}，{v.shape}")  # 打印qkv线性变换后的形状

        # 3. 计算缩放点积注意力
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. 将多个头的输出连接起来，并通过线性层映射回原始的维度
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. 返回最终的输出和注意力权重
        print(f"多头返回的out和attention权重的形状: {out.shape}，{attention.shape}")  # out和attention权重的形状
        return out, attention

    def split(self, tensor):
        """
        将输入的张量按头数拆分

        Parameters:
        - tensor: 输入张量，形状为 [batch_size, seq_len, d_model]

        Returns:
        - tensor: 拆分后的张量，形状为 [batch_size, n_head, seq_len, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        # 计算每个头的维度
        d_tensor = d_model // self.n_head
        # 将张量重塑为 [batch_size, seq_len, n_head, d_tensor] 然后转换维度
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        将拆分后的张量按头数合并

        Parameters:
        - tensor: 拆分后的张量，形状为 [batch_size, n_head, seq_len, d_tensor]

        Returns:
        - tensor: 合并后的张量，形状为 [batch_size, seq_len, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        # 逆转之前的转置操作，并将其重塑回原始形状
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
