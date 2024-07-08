import numpy as np
import torch.nn.functional as F
from getVCF_npy import get_SNP_groundtruth, get_CN_GroundTruth
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
from VAE import VAE
import torch
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

    def load_sample(self,data_path,flag = 'SNV'):
        self.flag = flag
        if flag == 'SNV':
            self.weight_path = 'weight_VAE/model_weights_SNV_Encode.pth'
            self.sample, _ = get_CN_GroundTruth(data_path)
        else:
            self.weight_path = 'weight_VAE/model_weights_CN.pth'
            self.sample,self.max_RD,self.min_RD = get_SNP_groundtruth(data_path)

    def loss_fn(self,y, x, mu, log_var):  # losses
        y = y[:, :, :, 4]
        y = y/(torch.max(y)-torch.min(y))
        recons_loss = F.mse_loss(y, x[:,:,:,4])  # MSE
        kld_loss = torch.mean(0.5 * torch.sum(mu ** 2 + torch.exp(log_var) - log_var - 1, 1), 0)  # 分布损失，正态分布与标准正态分布的KL散度
        answer = ((1-self.w)*recons_loss+ self.w * kld_loss)*1024
        return answer


    def trainning(self):
        cuda = True if torch.cuda.is_available() else False  # if cuda is useable
        code_size = (self.geno_channel, self.information_size, self.genomatic_length)  # size of sample (1,2048,5)
        vae = VAE(code_size, self.latent_dim)  # VAE
        if cuda:  #
            vae = vae.cuda()  #
        optimizer = torch.optim.Adagrad(vae.parameters(), lr=self.lr)
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
                if cuda:  # 如果使用cuda
                    geno_code = np.expand_dims(geno_code, axis=1)
                    geno_code = torch.tensor(geno_code)
                    geno_code = geno_code.cuda()
                vae.train()
                optimizer.zero_grad()
                y, x, mu, log_var = vae(geno_code)
                loss = self.loss_fn(y, x, mu, log_var)  # loss
                loss.backward()  # backward
                optimizer.step()  # update the weight
                total_loss += loss.item()  # total_loss of all batch
                pbar.set_postfix(**{"Loss": loss.item()})  #
                pbar.update(self.batch_size)  #
            pbar.close()  #
            avg_loss = total_loss / (len(self.sample) // self.batch_size)
            loss_line.append(avg_loss)
            print("total_loss:%.4f" %
                  (avg_loss))
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
                y_1 = self.debeding_RD(y,i,4)
                if x_total==[]:
                    x_total = x_1
                    y_total = y_1
                else:
                    x_total = np.append(x_total,x_1)
                    y_total = np.append(y_total,y_1)
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