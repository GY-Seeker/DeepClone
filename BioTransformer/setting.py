import numpy as np  # 导入 numpy 库
import torch  # 导入 torch 库
import torch.nn as nn  # 导入 torch.nn 库
d_k = 16  # K(=Q) 维度
d_v = 16  # V 维度
# 定义多头自注意力类
d_embedding = 512  # Embedding 的维度
n_heads = 4 # Multi-Head Attention 中头的个数
src_len = 1024
tgt_len = 1024
tgt_vocab = 1024
src_vocab = 1024
batch_sizes = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")