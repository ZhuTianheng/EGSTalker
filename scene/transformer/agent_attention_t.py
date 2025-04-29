import math
import torch
from torch import nn

class AgentScaleDotProductAttention(nn.Module):
    """
    改进后的缩放点积注意力机制，直接处理 [batch_size, seq_len, d_model] 的输入。
    """

    def __init__(self, d_model, num_heads, agent_num):
        """
        初始化 AgentScaleDotProductAttention 层
        
        Parameters:
        - d_model: 模型维度
        - num_heads: 注意力头的数量
        - agent_num: 代理 tokens 的数量
        """
        super(AgentScaleDotProductAttention, self).__init__()
        self.num_heads = num_heads
        self.agent_num = agent_num
        self.head_dim = d_model // num_heads  # 每个头的维度

        # 输出投影层
        self.proj = nn.Linear(d_model, d_model)
        # 线性变换层用于降维 attention
        self.attention_proj_layer = nn.Linear(agent_num, d_model)
        # 自适应平均池化，用于生成代理 tokens
        self.adaptive_pool = nn.AdaptiveAvgPool1d(agent_num)
        
        # Softmax 用于计算注意力权重
        self.softmax = nn.Softmax(dim=-1)
        
        # 代理 token 的线性变换
        self.agent_linear = nn.Linear(d_model, d_model)
        
        # Dropout 层
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        """
        前向传播，计算代理注意力
        
        Parameters:
        - q: 查询张量，形状为 [batch_size, seq_len, d_model]
        - k: 键张量，形状为 [batch_size, seq_len, d_model]
        - v: 值张量，形状为 [batch_size, seq_len, d_model]
        - mask: 可选的遮罩，形状为 [batch_size, seq_len, seq_len]
        
        Returns:
        - 加权后的值，形状为 [batch_size, seq_len, d_model]
        - 注意力权重，形状为 [batch_size, num_heads, seq_len, agent_num]
        """
        b, seq_len, _ = q.size()

        # 1. 生成代理 tokens
        agent_tokens = self.agent_linear(q[:, :self.agent_num, :])  # [batch_size, agent_num, d_model]

        # 2. 代理聚合：计算代理 token 与键的注意力
        k_t = k.transpose(-2, -1)  # [batch_size, seq_len, d_model] -> [batch_size, d_model, seq_len]
        agent_attn_scores = (agent_tokens @ k_t) / math.sqrt(self.head_dim)  # [batch_size, agent_num, seq_len]
        
        if mask is not None:
            agent_attn_scores = agent_attn_scores.masked_fill(mask.unsqueeze(1) == 0, -10000)
        agent_attn_scores = self.softmax(agent_attn_scores)  # [batch_size, agent_num, seq_len]
        
        # 使用代理注意力权重加权值
        agent_values = agent_attn_scores @ v  # [batch_size, agent_num, d_model]

        # 3. 代理广播：计算查询与代理 token 的注意力
        agent_t = agent_tokens.transpose(-2, -1)  # [batch_size, agent_num, d_model] -> [batch_size, d_model, agent_num]
        q_attn_scores = (q @ agent_t) / math.sqrt(self.head_dim)  # [batch_size, seq_len, agent_num]
        q_attn_scores = self.softmax(q_attn_scores)  # [batch_size, seq_len, agent_num]

        # 4. 最终的加权输出
        out = q_attn_scores @ agent_values  # [batch_size, seq_len, d_model]

        # 5. 输出投影
        attention = self.split(out)  # [batch_size, seq_len, d_model]
        #print(f"a-t输出的out,attention的形状: {out.shape}，{attention.shape}")
        return out, attention
    
    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.num_heads
        tensor = tensor.view(batch_size, length, self.num_heads, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor
