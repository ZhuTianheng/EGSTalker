import torch.nn as nn
import torch
import math

from scene.transformer.layer_norm import LayerNorm
from scene.transformer.position_wise_feed_forward import PositionwiseFeedForward
from scene.transformer.agent_attention_t import AgentScaleDotProductAttention


# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        #output_pe = self.pe[:, :x.size(1), :]
        #print(output_pe.shape)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Spatial_Audio_Attention_Layer(nn.Module):
    def __init__(self, args):
        super(Spatial_Audio_Attention_Layer, self).__init__()
        self.args = args
        
        self.enc_dec_attention = AgentScaleDotProductAttention(d_model=self.args.d_model, num_heads=self.args.n_head, agent_num=self.args.agent_num)
        
        self.norm1 = LayerNorm(d_model=self.args.d_model)
        self.dropout1 = nn.Dropout(p=self.args.drop_prob)  

        
        self.ffn = PositionwiseFeedForward(d_model=self.args.d_model, hidden=self.args.ffn_hidden, drop_prob=self.args.drop_prob)
        
        self.norm2 = LayerNorm(d_model=self.args.d_model)
        self.dropout2 = nn.Dropout(p=self.args.drop_prob)

        # Periodic Positional Encoding (PPE)
        self.PPE = PeriodicPositionalEncoding(d_model=self.args.d_model, period=25, max_seq_len=600)
        
    def forward(self, x, enc_source):
        """
        前向传播函数

        Parameters:
        - x: 当前层的输入，通常是来自前一层的输出，形状为 [batch_size, seq_len, d_model]
        - enc_source: 编码器输出的源序列，形状为 [batch_size, seq_len, d_model]

        Returns:
        - x: 经过注意力计算后的输出，形状为 [batch_size, seq_len, d_model]
        - att: 当前层的注意力权重
        """
        
        enc_source = self.PPE(enc_source)
        _x = x  
        
        x, att = self.enc_dec_attention(q=x, k=enc_source, v=enc_source, mask=None)
        x = self.dropout1(x)  
        x = self.norm1(x + _x)  
        _x = x  
        x = self.ffn(x)

        
        x = self.dropout2(x)  
        x = self.norm2(x + _x)  

        return x, att  


class Spatial_Audio_Attention_Module(nn.Module):
    def __init__(self, args):
        super(Spatial_Audio_Attention_Module, self).__init__()
        self.args = args
        
        # 构造多个 Spatial_Audio_Attention_Layer 层，形成堆叠结构
        self.layers = nn.ModuleList([Spatial_Audio_Attention_Layer(args) for _ in range(args.n_layer)])
        
    def forward(self, x, enc_source):
        """
        前向传播函数

        Parameters:
        - x: 输入张量，形状为 [batch_size, seq_len, d_model]
        - enc_source: 编码器的输出，形状为 [batch_size, seq_len, d_model]

        Returns:
        - x: 经过多个注意力层处理后的输出
        - attention: 每一层的注意力权重平均值，形状为 [batch_size, n_layer, 1, seq_len]
        """
        attention = []  # 用于保存每层的注意力权重
        for layer in self.layers:
            
            x, att = layer(x, enc_source)  
            
            attention.append(att.mean(dim=1).unsqueeze(dim=1))  

        
        attention = torch.cat(attention, dim=1) 
        
        return x, attention  




