from BioTransformer.setting import *


# 定义缩放点积注意力类
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()


    def forward(self, Q, K, V, attn_mask):
        # ------------------------- 维度信息 --------------------------------
        # Q K V [batch_size, n_heads, len_q/k/v, dim_q=k/v] (dim_q=dim_k)
        # attn_mask [batch_size, n_heads, len_q, len_k]
        # ----------------------------------------------------------------
        # 计算注意力分数（原始权重）[batch_size，n_heads，len_q，len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores =scores / np.sqrt(d_k)
        # ------------------------- 维度信息 --------------------------------
        # scores [batch_size, n_heads, len_q, len_k]
        # -----------------------------------------------------------------
        # 使用注意力掩码，将 attn_mask 中值为 1 的位置的权重替换为极小值
        # ------------------------- 维度信息 --------------------------------
        # attn_mask [batch_size, n_heads, len_q, len_k], 形状和 scores 相同
        # -----------------------------------------------------------------
        scores.masked_fill_(attn_mask, -1e9)
        # 对注意力分数进行 softmax 归一化
        weights = nn.Softmax(dim=-1)(scores)
        # ------------------------- 维度信息 --------------------------------
        # weights [batch_size, n_heads, len_q, len_k], 形状和 scores 相同
        # -----------------------------------------------------------------
        # 计算上下文向量（也就是注意力的输出）, 是上下文信息的紧凑表示
        context = torch.matmul(weights, V)
        # ------------------------- 维度信息 --------------------------------
        # context [batch_size, n_heads, len_q, dim_v]
        # -----------------------------------------------------------------
        return context, weights  # 返回上下文向量和注意力分数

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_embedding, d_embedding*d_k * n_heads) # Q的线性变换层
        self.W_K = nn.Linear(d_embedding, d_embedding*d_k * n_heads) # K的线性变换层
        self.W_V = nn.Linear(d_embedding, d_embedding*d_v * n_heads) # V的线性变换层
        self.linear = nn.Linear(n_heads * d_v * d_embedding, d_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)
    def forward(self, Q, K, V, attn_mask):
        #------------------------- 维度信息 --------------------------------
        # Q K V [batch_size, len_q/k/v, embedding_dim]
        #-----------------------------------------------------------------
        residual, batch_size = Q, Q.size(0) # 保留残差连接
        residual = residual.unsqueeze(1)
        # 将输入进行线性变换和重塑，以便后续处理
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)
        #------------------------- 维度信息 --------------------------------
        # q_s k_s v_s: [batch_size, n_heads, len_q/k/v, d_q=k/v]
        #-----------------------------------------------------------------
        # 将注意力掩码复制到多头 attn_mask: [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        #------------------------- 维度信息 --------------------------------
        # attn_mask [batch_size, n_heads, len_q, len_k]
        #-----------------------------------------------------------------
        # 使用缩放点积注意力计算上下文和注意力权重
        context, weights = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        #------------------------- 维度信息 --------------------------------
        # context [batch_size, n_heads, len_q, dim_v]
        # weights [batch_size, n_heads, len_q, len_k]
        #-----------------------------------------------------------------
        # 通过调整维度将多个头的上下文向量连接在一起
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,d_embedding*n_heads * d_v)
        # context = context.squeeze()
        #------------------------- 维度信息 --------------------------------
        # context [batch_size, len_q, n_heads * dim_v]
        #-----------------------------------------------------------------
        # 用一个线性层把连接后的多头自注意力结果转换，原始地嵌入维度
        output = self.linear(context)
        #------------------------- 维度信息 --------------------------------
        # output [batch_size, len_q, embedding_dim]
        #-----------------------------------------------------------------
        # 与输入 (Q) 进行残差链接，并进行层归一化后输出
        output = self.layer_norm(output + residual)
        #------------------------- 维度信息 --------------------------------
        # output [batch_size, len_q, embedding_dim]
        #-----------------------------------------------------------------
        return output, weights # 返回层归一化的输出和注意力权重