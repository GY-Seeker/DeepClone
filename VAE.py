import torch
import torch.nn as nn

class VAE(nn.Module):  # VAE
    def __init__(self, code_size, latent_dim):  # 初始化方法
        super(VAE, self).__init__()  # 继承初始化方法
        self.in_channel, self.img_h, self.img_w = code_size
        self.h = 1  # 6 convs，hight of feature1
        self.w = 1  # 6 convs，width of feature 1
        hw = self.h * self.w  # 最终特征层的尺寸hxw
        self.latent_dim = latent_dim  # size of Z
        self.hidden_dims = [32, 64, 128, 256, 512, 1024] # channel
        # Encoder
        self.encode_block_1 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.hidden_dims[0], (8,2), (4,1),(2,0)),  # add conv
            nn.BatchNorm2d(self.hidden_dims[0]),  # add bn
            nn.LeakyReLU() )  # output  channl,32,512,4
        self.encode_block_2 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[0], self.hidden_dims[1], (8, 2), (4, 1), (2, 0)),  # add conv
            nn.BatchNorm2d(self.hidden_dims[1]),  # add bn
            nn.LeakyReLU())  # output  channl,64,128,3
        self.encode_block_3 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[1], self.hidden_dims[2], (8, 2), (4, 1), (2, 0)),  # add conv
            nn.BatchNorm2d(self.hidden_dims[2]),  # add bn
            nn.LeakyReLU())  # output  channl,128,32,2
        self.encode_block_4 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[2], self.hidden_dims[3], (8, 2), (4, 1), (2, 0)),  # add conv
            nn.BatchNorm2d(self.hidden_dims[3]),  # add bn
            nn.LeakyReLU())  # output  channl,256,8,1
        self.encode_block_5 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[3], self.hidden_dims[4], (8, 1), (4, 1), (2,0)),  # add conv
            nn.BatchNorm2d(self.hidden_dims[4]),  # add bn
            nn.LeakyReLU())  # output  channl,512,2,1
        self.encode_block_6 = nn.Sequential(
            nn.Conv2d(self.hidden_dims[4], self.hidden_dims[5], (4, 1), (2, 1), (1,0)),  # add conv
            nn.BatchNorm2d(self.hidden_dims[5]),  #
            nn.LeakyReLU())  # output  channl,1024,1,1
        self.in_channel =self.hidden_dims[5]

        # 解码器decoder模型结构
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)  # linaer
        self.fc_var = nn.Linear(self.hidden_dims[-1] * hw, self.latent_dim)  # linear

        # 开始构建解码器Decoder
        self.decoder_input = nn.Linear(self.latent_dim, self.hidden_dims[-1] * hw)  # linaer，change to feature
        self.hidden_dims.reverse()

        self.decode_block_1 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[0], self.hidden_dims[1], (4,1), (2,1), (1,0),0),  # add transconv
            nn.BatchNorm2d(self.hidden_dims[1]),  # add bn
            nn.LeakyReLU())  # add leakyrelu
        self.decode_block_2 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[1], self.hidden_dims[2], (8,1), (2,1), (1,0), 0),  # add transconv
            nn.BatchNorm2d(self.hidden_dims[2]),  # add bn
            nn.LeakyReLU())  # add leakyrelu
        self.decode_block_3 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[2], self.hidden_dims[3], (8, 2), (4, 1),(2,0), 0),
            nn.BatchNorm2d(self.hidden_dims[3]),  # add bn
            nn.LeakyReLU())  # add leakyrelu
        self.decode_block_4 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[3], self.hidden_dims[4], (8, 2), (4, 1),(2,0) , 0),
            nn.BatchNorm2d(self.hidden_dims[4]),  # add bn
            nn.LeakyReLU())  # add leakyrelu
        self.decode_block_5 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[4], self.hidden_dims[5], (8, 2), (4, 1),(2,0) , 0),
            nn.BatchNorm2d(self.hidden_dims[5]),  # add bn
            nn.LeakyReLU()) # add leakyrelu
        self.decode_block_6 = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dims[5], self.hidden_dims[-1], (8, 2), (4, 1),(2,0) , 0),  # add transconv
            nn.BatchNorm2d(self.hidden_dims[-1]),  # add bn
            nn.LeakyReLU(),  # add leakyrelu
            nn.Conv2d(self.hidden_dims[-1], code_size[0], 1, 1, 0),
            nn.Tanh())  # add tanh

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
        result = self.encode(x)  # Encoder,(n,1,32,32)-->(n,512,1,1)
        result = torch.flatten(result, 1)
        mu = self.fc_mu(result)  # calculate mu,(n,512)-->(n,128)
        log_var = self.fc_var(result)  # log(var),(n,512)-->(n,128)
        return mu, log_var,result  #

    def decode(self, z):  # 定义解码过程
        y = self.decoder_input(z).view(-1, self.hidden_dims[0], self.h,
                                       self.w)  # (n,128)-->(n,512)-->(n,512,1,1)
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

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # std
        eps = torch.randn_like(std)  # (n,128)
        return mu + eps * std

    def forward(self, x):  # 前传函数
        mu, log_var,result = self.encode_pre(x)  # average mu log_var
        y = self.decode(result)  # output sample Y
        return [y, x, mu, log_var]

    def forward_pre(self,x):   # forward
        mu, log_var,result = self.encode_pre(x)
        z = self.reparameterize(mu, log_var)
        y = self.decode(z)
        return y

