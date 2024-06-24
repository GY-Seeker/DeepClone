from BioTransformer.setting import *
from BioTransformer.Tool import get_sin_enc_table,get_attn_pad_mask,get_attn_subsequent_mask
from BioTransformer.Attentions import MultiHeadAttention
from BioTransformer.Forward import PoswiseFeedForwardNet

#  定义解码器类
n_layers = 4  # 设置 Decoder 的层数
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Linear(src_len, d_embedding) # 词嵌入层
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)]) # 叠加多层
    def forward(self, dec_inputs, enc_outputs):
        #------------------------- 维度信息 --------------------------------
        # dec_inputs 的维度是 [batch_size, target_len]
        # enc_inputs 的维度是 [batch_size, source_len]
        # enc_outputs 的维度是 [batch_size, source_len, embedding_dim]
        #-----------------------------------------------------------------
        # 创建一个从 1 到 source_len 的位置索引序列
        # pos_indices = torch.arange(1, dec_inputs.size(1) + 1).unsqueeze(0).to(dec_inputs)
        #------------------------- 维度信息 --------------------------------
        # pos_indices 的维度是 [1, target_len]
        #-----------------------------------------------------------------
        # 对输入进行词嵌入和位置嵌入相加
        dec_outputs = self.tgt_emb(dec_inputs) # + self.pos_emb(pos_indices)
        #------------------------- 维度信息 --------------------------------
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
         #-----------------------------------------------------------------
        # 生成解码器自注意力掩码和解码器 - 编码器注意力掩码
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_outputs, dec_outputs) # 填充位掩码
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_outputs) # 后续位掩码
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask \
                                       + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_outputs, dec_outputs) # 解码器 - 编码器掩码
        #------------------------- 维度信息 --------------------------------
        # dec_self_attn_pad_mask 的维度是 [batch_size, target_len, target_len]
        # dec_self_attn_subsequent_mask 的维度是 [batch_size, target_len, target_len]
        # dec_self_attn_mask 的维度是 [batch_size, target_len, target_len]
        # dec_enc_attn_mask 的维度是 [batch_size, target_len, source_len]
         #-----------------------------------------------------------------
        dec_self_attns, dec_enc_attns = [], [] # 初始化 dec_self_attns, dec_enc_attns
        # 通过解码器层 [batch_size, seq_len, embedding_dim]
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs,
                                               dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        #------------------------- 维度信息 --------------------------------
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_self_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, target_len, target_len]
        # dec_enc_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, target_len, source_len]
        #-----------------------------------------------------------------
        # 返回解码器输出，解码器自注意力和解码器 - 编码器注意力权重
        return dec_outputs, dec_self_attns, dec_enc_attns


# 定义解码器层类
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention() # 多头自注意力层
        self.dec_enc_attn = MultiHeadAttention()  # 多头自注意力层，连接编码器和解码器
        self.pos_ffn = PoswiseFeedForwardNet() # 位置前馈神经网络层
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        #------------------------- 维度信息 --------------------------------
        # dec_inputs 的维度是 [batch_size, target_len, embedding_dim]
        # enc_outputs 的维度是 [batch_size, source_len, embedding_dim]
        # dec_self_attn_mask 的维度是 [batch_size, target_len, target_len]
        # dec_enc_attn_mask 的维度是 [batch_size, target_len, source_len]
        #-----------------------------------------------------------------
        # 将相同的 Q，K，V 输入多头自注意力层
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs, dec_self_attn_mask)
        dec_outputs = dec_outputs.squeeze()
        #------------------------- 维度信息 --------------------------------
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_self_attn 的维度是 [batch_size, n_heads, target_len, target_len]
        #-----------------------------------------------------------------
        # 将解码器输出和编码器输出输入多头自注意力层
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs, dec_enc_attn_mask)
        #------------------------- 维度信息 --------------------------------
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_enc_attn 的维度是 [batch_size, n_heads, target_len, source_len]
        #-----------------------------------------------------------------
        # 输入位置前馈神经网络层
        dec_outputs = self.pos_ffn(dec_outputs)
        dec_outputs = dec_outputs.squeeze()
        #------------------------- 维度信息 --------------------------------
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_self_attn 的维度是 [batch_size, n_heads, target_len, target_len]
        # dec_enc_attn 的维度是 [batch_size, n_heads, target_len, source_len]
        #-----------------------------------------------------------------
        # 返回解码器层输出，每层的自注意力和解 - 编码器注意力权重
        return dec_outputs, dec_self_attn, dec_enc_attn