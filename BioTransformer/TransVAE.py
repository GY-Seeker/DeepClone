import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from getVCF_npy import merge,merge_SNV
import matplotlib.pyplot as plt
import scipy

class VAE_Encode(nn.Module):  # 定义VAE模型
    # 卷积核考虑？
    def __init__(self, code_size, latent_dim):  # 初始化方法
        super(VAE_Encode, self).__init__()  # 继承初始化方法
        self.in_channel, self.img_h, self.img_w = code_size  # 由输入图片形状得到图片通道数C、图片高度H、图片宽度W
        self.h = 1  # 经过6次卷积后，最终特征层高度1
        self.w = 1  # 经过6次卷积后，最终特征层宽度1
        hw = self.h * self.w  # 最终特征层的尺寸hxw
        self.latent_dim = latent_dim  # 采样变量Z的长度
        self.hidden_dims = [32, 64, 128, 256, 512, 1024] # 特征层通道数列表
        # 开始构建编码器Encoder
        layers = []  # 用于存放模型结构
        self.encode_block_1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.hidden_dims[0], (8,2), (4,1),(2,0)),  # 添加conv
            nn.BatchNorm2d(self.hidden_dims[0]),  # 添加bn
            nn.LeakyReLU() )  # 输出 channl,32,512,4
        self.encode_block_2 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[0], self.hidden_dims[1], (8, 2), (4, 1), (2, 0)),  # 添加conv
            nn.BatchNorm2d(self.hidden_dims[1]),  # 添加bn
            nn.LeakyReLU())  # 输出 channl,64,128,3
        self.encode_block_3 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[1], self.hidden_dims[2], (8, 2), (4, 1), (2, 0)),  # 添加conv
            nn.BatchNorm2d(self.hidden_dims[2]),  # 添加bn
            nn.LeakyReLU())  # 输出 channl,128,32,2
        self.encode_block_4 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[2], self.hidden_dims[3], (8, 2), (4, 1), (2, 0)),  # 添加conv
            nn.BatchNorm2d(self.hidden_dims[3]),  # 添加bn
            nn.LeakyReLU())  # 输出 channl,256,8,1
        self.encode_block_5 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[3], self.hidden_dims[4], (8, 1), (4, 1), (2,0)),  # 添加conv
            nn.BatchNorm2d(self.hidden_dims[4]),  # 添加bn
            nn.LeakyReLU())  # 输出 channl,512,2,1
        self.encode_block_6 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[4], self.hidden_dims[5], (4, 1), (2, 1), (1,0)),  # 添加conv
            nn.BatchNorm2d(self.hidden_dims[5]),  #
            nn.LeakyReLU())  # 输出 channl,1024,1,1
        self.in_channel =self.hidden_dims[5]

        # 解码器decoder模型结构
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)  # linaer，将特征向量转化为分布均值mu
        self.fc_var = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)  # linear，将特征向量转化为分布方差的对数log(var)

        # 开始构建解码器Decoder
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * hw)  # linaer，将采样变量Z转化为特征向量
        self.hidden_dims.reverse()

        self.decode_block_1 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[0], self.hidden_dims[1], (4,1), (2,1), (1,0),0),  # 添加transconv
            nn.BatchNorm2d(self.hidden_dims[1]),  # 添加bn
            nn.LeakyReLU())  # 添加leakyrelu
        self.decode_block_2 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[1], self.hidden_dims[2], (8,1), (2,1), (1,0), 0),  # 添加transconv
            nn.BatchNorm2d(self.hidden_dims[2]),  # 添加bn
            nn.LeakyReLU())  # 添加leakyrelu
        self.decode_block_3 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[2], self.hidden_dims[3], (8, 2), (4, 1),(2,0), 0),
            nn.BatchNorm2d(self.hidden_dims[3]),  # 添加bn
            nn.LeakyReLU())  # 添加leakyrelu
        self.decode_block_4 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[3], self.hidden_dims[4], (8, 2), (4, 1),(2,0) , 0),
            nn.BatchNorm2d(self.hidden_dims[4]),  # 添加bn
            nn.LeakyReLU())  # 添加leakyrelu
        self.decode_block_5 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[4], self.hidden_dims[5], (8, 2), (4, 1),(2,0) , 0),
            nn.BatchNorm2d(self.hidden_dims[5]),  # 添加bn
            nn.LeakyReLU()) # 添加leakyrelu
        self.decode_block_6 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[5], self.hidden_dims[-1], (8, 2), (4, 1),(2,0) , 0),  # 添加transconv
            nn.BatchNorm2d(self.hidden_dims[-1]),  # 添加bn
            nn.LeakyReLU(),  # 添加leakyrelu
            nn.Conv2d(self.hidden_dims[-1], code_size[0], 1, 1, 0),
            nn.Tanh())  # 添加tanh

    def encode(self, x):  # 定义编码过程
        encode1 = self.encode_block_1(x)
        self.encode2 = self.encode_block_2(encode1)
        self.encode3 = self.encode_block_3(self.encode2)
        self.encode4 = self.encode_block_4(self.encode3)
        self.encode5 = self.encode_block_5(self.encode4)
        result = self.encode_block_6(self.encode5)
        result = torch.flatten(result, 1)  # 将特征层转化为特征向量,(n,512,1,1)-->(n,512)
        return result

    def forward(self, x):  # 前传函数
        result = self.encode(x)  # 经过编码过程，得到分布的均值mu和方差对数log_var
        return result  # 返回生成样本Y，输入样本X，分布均值mu，分布方差对数log_var


class VAE_Decode(nn.Module):  # 定义VAE模型
    # 卷积核考虑？
    def __init__(self, code_size, latent_dim):  # 初始化方法
        super(VAE_Decode, self).__init__()  # 继承初始化方法
        self.in_channel, self.img_h, self.img_w = code_size  # 由输入图片形状得到图片通道数C、图片高度H、图片宽度W
        self.h = 1  # 经过6次卷积后，最终特征层高度1
        self.w = 1  # 经过6次卷积后，最终特征层宽度1
        hw = self.h * self.w  # 最终特征层的尺寸hxw
        self.latent_dim = latent_dim  # 采样变量Z的长度
        self.hidden_dims = [32, 64, 128, 256, 512, 1024] # 特征层通道数列表
        # 开始构建编码器Encoder
        layers = []  # 用于存放模型结构
        self.encode_block_1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.hidden_dims[0], (8,2), (4,1),(2,0)),  # 添加conv
            nn.BatchNorm2d(self.hidden_dims[0]),  # 添加bn
            nn.LeakyReLU() )  # 输出 channl,32,512,4
        self.encode_block_2 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[0], self.hidden_dims[1], (8, 2), (4, 1), (2, 0)),  # 添加conv
            nn.BatchNorm2d(self.hidden_dims[1]),  # 添加bn
            nn.LeakyReLU())  # 输出 channl,64,128,3
        self.encode_block_3 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[1], self.hidden_dims[2], (8, 2), (4, 1), (2, 0)),  # 添加conv
            nn.BatchNorm2d(self.hidden_dims[2]),  # 添加bn
            nn.LeakyReLU())  # 输出 channl,128,32,2
        self.encode_block_4 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[2], self.hidden_dims[3], (8, 2), (4, 1), (2, 0)),  # 添加conv
            nn.BatchNorm2d(self.hidden_dims[3]),  # 添加bn
            nn.LeakyReLU())  # 输出 channl,256,8,1
        self.encode_block_5 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[3], self.hidden_dims[4], (8, 1), (4, 1), (2,0)),  # 添加conv
            nn.BatchNorm2d(self.hidden_dims[4]),  # 添加bn
            nn.LeakyReLU())  # 输出 channl,512,2,1
        self.encode_block_6 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[4], self.hidden_dims[5], (4, 1), (2, 1), (1,0)),  # 添加conv
            nn.BatchNorm2d(self.hidden_dims[5]),  #
            nn.LeakyReLU())  # 输出 channl,1024,1,1
        self.in_channel =self.hidden_dims[5]

        # 解码器decoder模型结构
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)  # linaer，将特征向量转化为分布均值mu
        self.fc_var = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)  # linear，将特征向量转化为分布方差的对数log(var)

        # 开始构建解码器Decoder
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * hw)  # linaer，将采样变量Z转化为特征向量
        self.hidden_dims.reverse()

        self.decode_block_1 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[0], self.hidden_dims[1], (4,1), (2,1), (1,0),0),  # 添加transconv
            nn.BatchNorm2d(self.hidden_dims[1]),  # 添加bn
            nn.LeakyReLU())  # 添加leakyrelu
        self.decode_block_2 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[1], self.hidden_dims[2], (8,1), (2,1), (1,0), 0),  # 添加transconv
            nn.BatchNorm2d(self.hidden_dims[2]),  # 添加bn
            nn.LeakyReLU())  # 添加leakyrelu
        self.decode_block_3 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[2], self.hidden_dims[3], (8, 2), (4, 1),(2,0), 0),
            nn.BatchNorm2d(self.hidden_dims[3]),  # 添加bn
            nn.LeakyReLU())  # 添加leakyrelu
        self.decode_block_4 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[3], self.hidden_dims[4], (8, 2), (4, 1),(2,0) , 0),
            nn.BatchNorm2d(self.hidden_dims[4]),  # 添加bn
            nn.LeakyReLU())  # 添加leakyrelu
        self.decode_block_5 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[4], self.hidden_dims[5], (8, 2), (4, 1),(2,0) , 0),
            nn.BatchNorm2d(self.hidden_dims[5]),  # 添加bn
            nn.LeakyReLU()) # 添加leakyrelu
        self.decode_block_6 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[5], self.hidden_dims[-1], (8, 2), (4, 1),(2,0) , 0),  # 添加transconv
            nn.BatchNorm2d(self.hidden_dims[-1]),  # 添加bn
            nn.LeakyReLU(),  # 添加leakyrelu
            nn.Conv2d(self.hidden_dims[-1], code_size[0], 1, 1, 0),
            nn.Tanh())  # 添加tanh

    def decode(self, z):  # 定义解码过程
        y = self.decoder_input(z).view(-1, self.hidden_dims[0], self.h,
                                       self.w)  # 将采样变量Z转化为特征向量，再转化为特征层,(n,128)-->(n,512)-->(n,512,1,1)
        decode1 = self.decode_block_1(y)
        decode2 = self.decode_block_2(decode1)
        decode3 = self.decode_block_3(decode2)
        decode4 = self.decode_block_4(decode3)
        decode5 = self.decode_block_5(decode4)
        result = self.decode_block_6(decode5)
        return result  # 返回生成样本Y

    def forward(self, voctor):  # 前传函数
        result = self.decode(voctor)  # 经过解码过程，得到生成样本Y
        return result  # 返回生成样本Y，输入样本X，分布均值mu，分布方差对数log_var