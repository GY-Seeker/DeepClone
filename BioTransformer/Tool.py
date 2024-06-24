from BioTransformer.setting import *
# 生成正弦位置编码表的函数，用于在 Transformer 中引入位置信息

def get_sin_enc_table(n_position, embedding_dim):
    #------------------------- 维度信息 --------------------------------
    # n_position: 输入序列的最大长度
    # embedding_dim: 词嵌入向量的维度
    #-----------------------------------------------------------------
    # 根据位置和维度信息，初始化正弦位置编码表
    sinusoid_table = np.zeros((n_position, embedding_dim))
    # 遍历所有位置和维度，计算角度值
    for pos_i in range(n_position):
        for hid_j in range(embedding_dim):
            angle = pos_i / np.power(10000, 2 * (hid_j // 2) / embedding_dim)
            sinusoid_table[pos_i, hid_j] = angle
    # 计算正弦和余弦值
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i 偶数维
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1 奇数维
    #------------------------- 维度信息 --------------------------------
    # sinusoid_table 的维度是 [n_position, embedding_dim]
    #----------------------------------------------------------------
    return torch.FloatTensor(sinusoid_table)  # 返回正弦位置编码表

# 定义填充注意力掩码函数
def get_attn_pad_mask(seq_q, seq_k):
    #------------------------- 维度信息 --------------------------------
    # seq_q 的维度是 [batch_size, len_q]
    # seq_k 的维度是 [batch_size, len_k]
    #-----------------------------------------------------------------
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # 生成布尔类型张量
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # <PAD>token 的编码值为 0
    #------------------------- 维度信息 --------------------------------
    # pad_attn_mask 的维度是 [batch_size，1，len_k]
    #-----------------------------------------------------------------
    # 变形为与注意力分数相同形状的张量
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    #------------------------- 维度信息 --------------------------------
    # pad_attn_mask 的维度是 [batch_size，len_q，len_k]
    #-----------------------------------------------------------------
    # pad_attn_mask = pad_attn_mask.cuda()
    return pad_attn_mask

# 生成后续注意力掩码的函数，用于在多头自注意力计算中忽略未来信息
def get_attn_subsequent_mask(seq):
    #------------------------- 维度信息 --------------------------------
    # seq 的维度是 [batch_size, seq_len(Q)=seq_len(K)]
    #-----------------------------------------------------------------
    # 获取输入序列的形状
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    #------------------------- 维度信息 --------------------------------
    # attn_shape 是一个一维张量 [batch_size, seq_len(Q), seq_len(K)]
    #-----------------------------------------------------------------
    # 使用 numpy 创建一个上三角矩阵（triu = triangle upper）
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    #------------------------- 维度信息 --------------------------------
    # subsequent_mask 的维度是 [batch_size, seq_len(Q), seq_len(K)]
    #-----------------------------------------------------------------
    # 将 numpy 数组转换为 PyTorch 张量，并将数据类型设置为 byte（布尔值）
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    #------------------------- 维度信息 --------------------------------
    # 返回的 subsequent_mask 的维度是 [batch_size, seq_len(Q), seq_len(K)]
    #-----------------------------------------------------------------
    subsequent_mask = subsequent_mask.to(device)
    return subsequent_mask # 返回后续位置的注意力掩码