import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from getVCF_npy import merge,merge_SNV
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import mean_squared_error
from torchvision import transforms, datasets
from torchvision.utils import save_image
from tqdm import tqdm


class VAE(nn.Module):  # 定义VAE模型
    # 卷积核考虑？
    def __init__(self, code_size, latent_dim):  # 初始化方法
        super(VAE, self).__init__()  # 继承初始化方法
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

    def encode_pre(self, x):
        result = self.encode(x)  # Encoder结构,(n,1,32,32)-->(n,512,1,1)
        result = torch.flatten(result, 1)
        mu = self.fc_mu(result)  # 计算分布均值mu,(n,512)-->(n,128)
        log_var = self.fc_var(result)  # 计算分布方差的对数log(var),(n,512)-->(n,128)
        return mu, log_var,result  # 返回分布的均值和方差对数

    def decode(self, z):  # 定义解码过程
        y = self.decoder_input(z).view(-1, self.hidden_dims[0], self.h,
                                       self.w)  # 将采样变量Z转化为特征向量，再转化为特征层,(n,128)-->(n,512)-->(n,512,1,1)
        decode1 = self.decode_block_1(y)
        # decode1 = torch.cat((decode1, self.encode5), dim=1)
        decode2 = self.decode_block_2(decode1)
        # decode2 = torch.cat((decode2, self.encode4), dim=1)
        decode3 = self.decode_block_3(decode2)
        # decode3 = torch.cat((decode3, self.encode3), dim=1)
        decode4 = self.decode_block_4(decode3)
        # decode4 = torch.cat((decode4, self.encode2), dim=1)
        decode5 = self.decode_block_5(decode4)
        result = self.decode_block_6(decode5)
        return result  # 返回生成样本Y

    def reparameterize(self, mu, log_var):  # 重参数技巧
        std = torch.exp(0.5 * log_var)  # 分布标准差std
        eps = torch.randn_like(std)  # 从标准正态分布中采样,(n,128)
        return mu + eps * std  # 返回对应正态分布中的采样值

    def forward(self, x):  # 前传函数
        mu, log_var,result = self.encode_pre(x)  # 经过编码过程，得到分布的均值mu和方差对数log_var
        # z = self.reparameterize(mu, log_var)  # 经过重参数技巧，得到分布采样变量Z
        y = self.decode(result)  # 经过解码过程，得到生成样本Y
        return [y, x, mu, log_var]  # 返回生成样本Y，输入样本X，分布均值mu，分布方差对数log_var

    def forward_pre(self,x):  # 定义生成过程
        mu, log_var,result = self.encode_pre(x)  # 经过编码过程，得到分布的均值mu和方差对数log_var
        z = self.reparameterize(mu, log_var)  # 经过重参数技巧，得到分布采样变量Z
        y = self.decode(z)  # 经过解码过程，得到生成样本Y
        return y  # 返回生成样本Y

class TrainPrediction:

    def __init__(self,total_epochs = 1000, batch_size = 64,
                 lr = 20e-3, w = 0.025,
                 information_size = 5, genomatic_length = 2048,
                 geno_channel = 1, latent_dim = 1024):
        self.total_epochs =total_epochs
        self.batch_size =batch_size
        self.lr =lr
        self.information_size =information_size
        self.genomatic_length = genomatic_length
        self.geno_channel = geno_channel
        self.latent_dim = latent_dim
        self.w=w

    def load_sample(self,flag = 0):
        self.flag = flag
        if flag == 0:
            self.sample,_ = merge_SNV()
            self.weight_path = 'weight_VAE/model_weights_SNV_Encode.pth'
        else:
            self.weight_path = 'weight_VAE/model_weights_CN.pth'
            self.sample,self.max_RD,self.min_RD = merge()

    def loss_fn(self,y, x, mu, log_var):  # 定义损失函数
        y = y[:, :, :, 4]
        y = y/(torch.max(y)-torch.min(y))
        recons_loss = F.mse_loss(y, x[:,:,:,4])  # 重建损失，MSE
        kld_loss = torch.mean(0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - log_var - 1, 1), 0)  # 分布损失，正态分布与标准正态分布的KL散度
        answer = ((1-self.w)*recons_loss+ self.w * kld_loss)*1024
        return answer#answer  # 最终损失由两部分组成，其中分布损失需要乘上一个系数w


    def trainning(self):
        cuda = True if torch.cuda.is_available() else False  # 如果cuda可用，则使用cuda
        code_size = (self.geno_channel, self.information_size, self.genomatic_length)  # 输入样本形状(1,2048,5)
        vae = VAE(code_size, self.latent_dim)  # 实例化VAE模型，传入输入样本形状与采样变量长度
        if cuda:  # 如果使用cuda
            vae = vae.cuda()  # 将模型加载到GPU
        optimizer = torch.optim.Adagrad(vae.parameters(), lr=self.lr)  # 使用Adam优化器
        # optimizer = torch.optim.Adam(vae.parameters(), lr=self.lr)
        loss_line = []
        # train loop
        last_loss = 200
        vae.load_state_dict(torch.load('weight_VAE/model_weights_SNV_Encode2.pth'))
        for epoch in range(self.total_epochs):  # 循环epoch
            total_loss = 0  # 记录总损失
            pbar = tqdm(total=len(self.sample), desc=f"Epoch {epoch + 1}/{self.total_epochs}", postfix=dict,
                        miniters=0.3)  # 设置当前epoch显示进度
            for i in range(0,len(self.sample),self.batch_size):  # 循环iter
                geno_code = self.sample[i:(i+self.batch_size),:,:]
                # truth = self.truth[i:(i+self.batch_size),:,:]
                if cuda:  # 如果使用cuda
                    geno_code = np.expand_dims(geno_code, axis=1)
                    geno_code = torch.tensor(geno_code)
                    geno_code = geno_code.cuda()
                    # truth = np.expand_dims(truth, axis=1)
                    # truth = torch.tensor(truth)
                    # truth = truth.cuda()
                    # geno_code = geno_code.cuda()  # 将训练数据加载到GPU
                vae.train()  # 模型开始训练
                optimizer.zero_grad()  # 模型清零梯度
                y, x, mu, log_var = vae(geno_code)  # 输入训练样本X，得到生成样本Y，输入样本X，分布均值mu，分布方差对数log_var
                loss = self.loss_fn(y, x, mu, log_var)  # 计算loss
                loss.backward()  # 反向传播，计算当前梯度
                optimizer.step()  # 根据梯度，更新网络参数
                total_loss += loss.item()  # 累计loss
                pbar.set_postfix(**{"Loss": loss.item()})  # 显示当前iter的loss
                pbar.update(self.batch_size)  # 步进长度
            pbar.close()  # 关闭当前epoch显示进度
            avg_loss = total_loss / (len(self.sample) // self.batch_size)
            loss_line.append(avg_loss)
            print("total_loss:%.4f" %
                  (avg_loss))  # 显示当前epoch训练完成后，模型的总损失
            if avg_loss<last_loss:
                torch.save(vae.state_dict(), 'weight_VAE/model_weights_' + str(epoch) + "_loss_" +str(avg_loss)+ ".pth")
                last_loss = avg_loss
        plt.plot(loss_line, marker='.')
        plt.show()

    def prediction(self):
        code_size = (self.geno_channel, self.information_size, self.genomatic_length)  # 输入样本形状(1,2048,5)
        vae = VAE(code_size, self.latent_dim)
        vae = vae.cuda()  # 将模型加载到GPU
        vae.load_state_dict(torch.load(self.weight_path))
        optimizer = torch.optim.SGD(vae.parameters(), lr=self.lr)
        x_total = []
        y_total = []
        for ii in range(0, len(self.sample), self.batch_size):
            geno_code = self.sample[ii:(ii+self.batch_size),:,:]
            geno_code = np.expand_dims(geno_code, axis=1)
            geno_code = torch.tensor(geno_code)
            geno_code = geno_code.cuda()
            vae.eval()  # 模型开始验证
            optimizer.zero_grad()
            y, x, mu, log_var = vae(geno_code)
            for i in range(y.shape[0]):
                x_1 = self.debeding_RD(x, i, 4)
                x_2 = self.debeding_RD(x,i,3)
                y_1 = self.debeding_RD(y,i,4)
                y_2 = self.debeding_RD(y, i, 3)
                if x_total==[]:
                    x_total = x_1#/(x_1+1)
                    y_total = y_1#/(y_1+1)
                else:
                    x_total = np.append(x_total,x_1)#/(x_1+1))
                    y_total = np.append(y_total,y_1)#/(y_1+1))
        return x_total,y_total

    def KL(self,p,q):
        p_one = p / (np.sum(p)+1)
        q_one = q / (np.sum(q)+1)
        KL = scipy.stats.entropy(p_one, q_one)
        return KL

    def debeding_RD(self,data,line,index,log_flag=1):
        data = data.cpu()
        DATA_LINE = data[line, 0, :, index].detach().numpy()
        DATA_LINE[DATA_LINE < 0] = 0
        if log_flag==0:
            DATA_LINE = np.power(10, DATA_LINE) - 1
        return DATA_LINE


import seaborn as sns
TP  = TrainPrediction()
TP.load_sample(0)
TP.trainning()
# x,y = TP.prediction()
# y=y/(np.max(y)-np.min(y))
# plt.plot(x[:12000],'o',linestyle='')
# plt.plot(y[:12000],'o',linestyle='')
# plt.show()
# x = x[x>0.01]
# y = y[y>0.01]
# # sns.relplot(x)
# ax = sns.displot(x, kind='kde')
# plt.show()
# ax = sns.displot(y, kind='kde')
# # sns.relplot(y)
# plt.show()

# file = open('3v3r.csv', mode='a', encoding='utf-8')
# file.write('Data Type')
# file.write(',')
# file.write('RD')
# file.write('\n')


"""
file.write('groundTruth')
file.write(',')
file.write(str(x_1[j]))
file.write('\n')
file.write('Result')
file.write(',')
file.write(str(y_1[j]))
file.write('\n')
"""