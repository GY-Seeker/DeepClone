from BioTransformer.setting import *
from BioTransformer.Encoder import Encoder
from BioTransformer.Decoder import Decoder
# 定义 Transformer 模型
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder() # 初始化编码器实例
        self.decoder = Decoder() # 初始化解码器实例
        # 定义线性投影层，将解码器输出转换为目标词汇表大小的概率分布
        self.projection = nn.Linear(d_embedding, tgt_vocab, bias=False)
    def forward(self, enc_inputs, dec_inputs):
        #------------------------- 维度信息 --------------------------------
        # enc_inputs 的维度是 [batch_size, source_seq_len]
        # dec_inputs 的维度是 [batch_size, target_seq_len]
        #-----------------------------------------------------------------
        # 将输入传递给编码器，并获取编码器输出和自注意力权重
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        #------------------------- 维度信息 --------------------------------
        # enc_outputs 的维度是 [batch_size, source_len, embedding_dim]
        # enc_self_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, src_seq_len, src_seq_len]
        #-----------------------------------------------------------------
        # 将编码器输出、解码器输入和编码器输入传递给解码器
        # 获取解码器输出、解码器自注意力权重和编码器 - 解码器注意力权重
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_outputs)
        #------------------------- 维度信息 --------------------------------
        # dec_outputs 的维度是 [batch_size, target_len, embedding_dim]
        # dec_self_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, tgt_seq_len, src_seq_len]
        # dec_enc_attns 是一个列表，每个元素的维度是 [batch_size, n_heads, tgt_seq_len, src_seq_len]
        #-----------------------------------------------------------------
        # 将解码器输出传递给投影层，生成目标词汇表大小的概率分布
        dec_logits = self.projection(dec_outputs)
        #------------------------- 维度信息 --------------------------------
        # dec_logits 的维度是 [batch_size, tgt_seq_len, tgt_vocab_size]
        #-----------------------------------------------------------------
        # 返回逻辑值 ( 原始预测结果 ), 编码器自注意力权重，解码器自注意力权重，解 - 编码器注意力权重
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns

    def save(self,test_loss):
        state_dict = self.state_dict()
        # 保存状态字典到文件
        torch.save(state_dict, 'weights_for_loss_'+test_loss+'.pth')

    def load(self):
        self.load_state_dict(torch.load('weights_for_loss.pth'))
        self.eval()