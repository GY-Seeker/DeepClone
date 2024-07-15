import copy

import torch

import getVCF_npy as gvcf
from BioTransformer.setting import *
from BioTransformer.TransVAE import VAE_Encode, VAE_Decode
# 定义 TranslationCorpus 类
class TranslationCorpus:
    def __init__(self,weight_path, information_size = 5,
                 genomatic_length = 2048, latent_dim = 1024,
                 geno_channel = 1
                 ):
        self.latent_dim = latent_dim
        self.information_size = information_size
        self.genomatic_length = genomatic_length
        self.geno_channel = geno_channel
        self.weight_path = weight_path

    def getInput(self,path1):
        file_list = gvcf.get_filelist(path1)
        input = gvcf.get_snv_npy(file_list)
        path2 = path1+'/groundTruth.npy'
        gt = np.load(path2)
        gt = gt.reshape([-1,2048])
        groundTruth = copy.deepcopy(input)
        gt = gt[:groundTruth.shape[0],:]
        groundTruth[:,:,4] = gt
        groundTruth = groundTruth.astype(np.float32)
        input = input.astype(np.float32)
        return input, groundTruth

    def getGroundTruth(self):
        pass


    # 定义创建词汇表的函数
    def VAEEncoding(self,data):
        data = torch.tensor(data)
        data = data.cuda()
        code_size = (self.geno_channel, self.information_size, self.genomatic_length)  # 输入样本形状(1,2048,5)
        vae = VAE_Encode(code_size, self.latent_dim)
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            vae = vae.cuda()
        vae.load_state_dict(torch.load(self.weight_path))
        vae.eval()
        Encode_data = vae(data)
        return Encode_data

    def VAEDecoding(self,data):
        data = torch.tensor(data)
        data = data.cuda()
        code_size = (self.geno_channel, self.information_size, self.genomatic_length)  # 输入样本形状(1,2048,5)
        vae = VAE_Decode(code_size, self.latent_dim)
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            vae = vae.cuda()
        vae.load_state_dict(torch.load(self.weight_path))
        vae.eval()
        Decode_data = vae(data)
        return Decode_data

    def make_Tensor(self,path1='D:/Pycharm/TranVAE/vcf/v1r1_B_60'):
        input_batch,target_batch = [],[]
        # 随机选择句子索引
        input, groundTruth = self.getInput(path1)
        for i in range(0,input.shape[0],batch_sizes):
            input_tmp = input[i:(i+batch_sizes),:,:]
            input_tmp = np.expand_dims(input_tmp, axis=1)
            input_data = self.VAEEncoding(input_tmp)
            target_tmp = groundTruth[i:(i+batch_sizes),:,:]
            target_tmp = np.expand_dims(target_tmp, axis=1)
            target_data = self.VAEEncoding(target_tmp)
            if type(input_batch)==type([]):
                input_batch = input_data
                target_batch = target_data
            else:
                input_batch = torch.cat((input_batch, input_data),0)
                target_batch = torch.cat((target_batch, target_data),0)
        # input_batch = torch.round(input_batch,decimals=3).mul(1000)
        # target_batch = torch.round(target_batch,decimals=3).mul(1000)
        # input_batch = input_batch.long()
        # target_batch = target_batch.long()
        return input_batch, input_batch, target_batch

    def get_org_file(self,path1='D:/Pycharm/TranVAE/vcf/v1r1_B_60'):
        input, groundTruth = self.getInput(path1)
        return input,groundTruth

    def make_SubClone(self,input):
        output_batch = []
        input =input.cpu()
        input = input.detach().numpy()
        for i in range(0, input.shape[0], batch_sizes):
            input_tmp = input[i:(i + batch_sizes), :]
            ouput_data = self.VAEDecoding(input_tmp)
            if type(output_batch)==type([]):
                output_batch = ouput_data
            else:
                output_batch = torch.cat((output_batch, ouput_data),0)
        output_batch = output_batch.squeeze()
        output_batch = output_batch.cpu()
        output_batch = output_batch.detach().numpy()
        return output_batch

