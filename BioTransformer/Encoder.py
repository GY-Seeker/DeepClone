from BioTransformer.setting import *
from BioTransformer.Tool import get_sin_enc_table,get_attn_pad_mask
from BioTransformer.Attentions import MultiHeadAttention
from BioTransformer.Forward import PoswiseFeedForwardNet

# 定义编码器类
n_layers = 6  # 设置 Encoder 的层数
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Linear(src_len, d_embedding) # 词嵌入层
        # self.pos_emb = nn.Linear.from_pretrained(
        #   get_sin_enc_table(src_len, d_embedding), freeze=True) # 位置嵌入层
        self.layers = nn.ModuleList(EncoderLayer() for _ in range(n_layers))# 编码器层数
    def forward(self, enc_inputs):
        #------------------------- 维度信息 --------------------------------
        # enc_inputs 的维度是 [batch_size, source_len]
        #-----------------------------------------------------------------
        # 创建一个从 1 到 source_len 的位置索引序列
        # pos_indices = torch.arange(1, enc_inputs.size(1) + 1).unsqueeze(0).to(enc_inputs)
        #------------------------- 维度信息 --------------------------------
        # pos_indices 的维度是 [1, source_len]
        #-----------------------------------------------------------------
        # 对输入进行词嵌入和位置嵌入相加 [batch_size, source_len，embedding_dim]
        # enc_outputs = enc_inputs
        enc_outputs = self.src_emb(enc_inputs) #+ self.pos_emb(pos_indices)
        #------------------------- 维度信息 --------------------------------
        # enc_outputs 的维度是 [batch_size, seq_len, embedding_dim]
        #-----------------------------------------------------------------
        # 生成自注意力掩码
        enc_self_attn_mask = get_attn_pad_mask(enc_outputs, enc_outputs)
        #------------------------- 维度信息 --------------------------------
        # enc_self_attn_mask 的维度是 [batch_size, len_q, len_k]
        #-----------------------------------------------------------------
        enc_self_attn_weights = [] # 初始化 enc_self_attn_weights
        # 通过编码器层 [batch_size, seq_len, embedding_dim]
        for layer in self.layers:
            enc_outputs, enc_self_attn_weight = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attn_weights.append(enc_self_attn_weight)
        #------------------------- 维度信息 --------------------------------
        # enc_outputs 的维度是 [batch_size, seq_len, embedding_dim] 维度与 enc_inputs 相同
        # enc_self_attn_weights 是一个列表，每个元素的维度是 [batch_size, n_heads, seq_len, seq_len]
        #-----------------------------------------------------------------
        return enc_outputs, enc_self_attn_weights # 返回编码器输出和编码器注意力权重

# 定义编码器层类
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention() # 多头自注意力层
        self.pos_ffn = PoswiseFeedForwardNet() # 位置前馈神经网络层
    def forward(self, enc_inputs, enc_self_attn_mask):
        #------------------------- 维度信息 --------------------------------
        # enc_inputs 的维度是 [batch_size, seq_len, embedding_dim]
        # enc_self_attn_mask 的维度是 [batch_size, seq_len, seq_len]
        #-----------------------------------------------------------------
        # 将相同的 Q，K，V 输入多头自注意力层 , 返回的 attn_weights 增加了头数
        enc_outputs, attn_weights = self.enc_self_attn(enc_inputs, enc_inputs,
                                               enc_inputs, enc_self_attn_mask)
        #------------------------- 维度信息 --------------------------------
        # enc_outputs 的维度是 [batch_size, seq_len, embedding_dim]
        # attn_weights 的维度是 [batch_size, n_heads, seq_len, seq_len]
        # 将多头自注意力 outputs 输入位置前馈神经网络层
        enc_outputs = self.pos_ffn(enc_outputs) # 维度与 enc_inputs 相同
        enc_outputs = enc_outputs.squeeze()
        #------------------------- 维度信息 --------------------------------
        # enc_outputs 的维度是 [batch_size, seq_len, embedding_dim]
        #-----------------------------------------------------------------
        return enc_outputs, attn_weights # 返回编码器输出和每层编码器注意力权重