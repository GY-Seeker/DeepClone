import copy

import numpy as np
import os

def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
                Filelist.append(os.path.join(home, filename))
    return Filelist

def transformer_CN_2_matrix(data_all_chrom):
    data_all = []
    for data in data_all_chrom:
        start_position = data[1,:]
        end_position = data[2,:]
        RD = data[3,:]+1
        start_position_sin = np.sin(np.deg2rad(start_position/100000))+1
        start_position_cos = np.cos(np.deg2rad(start_position / 100000))+1
        end_position_sin = np.sin(np.deg2rad(end_position / 100000))+1
        end_position_cos = np.cos(np.deg2rad(end_position / 100000))+1
        RD = np.log10(RD)
        tmp_array = np.array([start_position_sin,start_position_cos,end_position_sin,end_position_cos,RD])
        tmp_array = tmp_array.T
        data_all.append(tmp_array)
    data_all = np.array(data_all)
    maxx,minn = np.max(data_all[:,:,4]),np.min(data_all[:,:,4])
    data_all[:,:,4] = ((data_all[:,:,4]-minn)/(maxx-minn+1))*2
    data_all = data_all/2
    return data_all,maxx,minn

def transformer_SNV_2_matrix(data_all_chrom):
    data_all = []
    for data in data_all_chrom:
        start_position = data[1, :]
        AD = data[2, :]
        RD = data[5, :]
        start_position_sin = np.sin(np.deg2rad(start_position / 100000))+1
        start_position_cos = np.cos(np.deg2rad(start_position / 100000))+1
        VCF = AD/(RD+1)
        RD = np.log10(RD+1)
        AD = np.log10(AD+1)
        tmp_array = np.array([start_position_sin, start_position_cos,RD,AD,VCF])
        tmp_array = tmp_array.T
        data_all.append(tmp_array)
    data_all = np.array(data_all)
    maxx, minn = np.max(data_all[:, :, 2]), np.min(data_all[:, :, 2])

    data_all[:, :, 2] = ((data_all[:, :, 2] - np.min(data_all[:, :, 2])) / (
                np.max(data_all[:, :, 2]) - np.min(data_all[:, :, 2])+1))*2
    data_all[:, :, 3] = ((data_all[:, :, 3] - np.min(data_all[:, :, 3])) / (
            np.max(data_all[:, :, 3]) - np.min(data_all[:, :, 3])+1)) * 2
    data_all[:, :, 4] = data_all[:, :, 4]*2
    data_all = data_all / 2
    return data_all,maxx, minn

def get_CN_GroundTruth(path1):
    file_list = get_filelist(path1)
    sample = []
    maxx = 0
    minn = 1000
    for file_path in file_list:
        if ("vcf.npy" in file_path) or ("CN.npy" in file_path):
            data_all_chrom = np.load(file_path)
            matrix_all_chrom,maxs,mins = transformer_CN_2_matrix(data_all_chrom)
            maxx = max(maxx,maxs)
            minn = min(minn,mins)
            if sample == []:
                sample = matrix_all_chrom
            else:
                sample = np.concatenate((sample,matrix_all_chrom),axis=0)
    return sample,maxx,minn

def get_snv_npy(file_list):
    sample = []
    maxx = 0
    minn = 1000
    for file_path in file_list:
        if ("vcf2.npy" in file_path) or ("SNV.npy" in file_path):
            data_all_chrom = np.load(file_path)
            matrix_all_chrom,maxs, mins  = transformer_SNV_2_matrix(data_all_chrom)
            maxx = max(maxx, maxs)
            minn = min(minn, mins)
            if sample == []:
                sample = matrix_all_chrom
            else:
                sample = np.concatenate((sample, matrix_all_chrom), axis=0)
    return sample

def get_SNP_groundtruth(path1):
    file_list = get_filelist(path1)
    input = get_snv_npy(file_list)
    path2 = path1 + '/groundTruth.npy'
    gt = np.load(path2)
    gt = gt.reshape([-1, 2048])
    groundTruth = copy.deepcopy(input)
    groundTruth[:, :, 4] = gt
    groundTruth = groundTruth.astype(np.float32)
    input = input.astype(np.float32)
    return input, groundTruth

def merge():
    path1 = 'D:/Pycharm/TranVAE/vcf/v1r1_normal_60'
    sample1,maxx,minn = get_CN_GroundTruth(path1)
    path2 = 'D:/Pycharm/TranVAE/vcf/v1r2_normal_60'
    sample2,maxx,minn  = get_CN_GroundTruth(path2)
    sample = np.concatenate((sample1, sample2), axis=0)
    path3 = 'D:/Pycharm/TranVAE/vcf/v1r3_normal_60'
    sample3,maxx,minn = get_CN_GroundTruth(path3)
    sample = np.concatenate((sample, sample3), axis=0)
    path3 = 'D:/Pycharm/TranVAE/vcf/v2r1_normal_60'
    sample4,maxx,minn = get_CN_GroundTruth(path3)
    sample = np.concatenate((sample, sample4), axis=0)
    path3 = 'D:/Pycharm/TranVAE/vcf/v2r3_normal_60'
    sample5,maxx,minn = get_CN_GroundTruth(path3)
    sample = np.concatenate((sample, sample5), axis=0)
    path3 = 'D:/Pycharm/TranVAE/vcf/v3r3_normal_60'
    sample6,maxx,minn = get_CN_GroundTruth(path3)
    sample = np.concatenate((sample, sample6), axis=0)
    sample = sample.astype(np.float32)
    return sample,maxx,minn

def merge_SNV():
    path1 = 'D:/Pycharm/TranVAE/vcf/v1r1_B_60'
    # file_list = get_filelist(path1)
    sample1,truth1 = get_SNP_groundtruth(path1)
    path2 = 'D:/Pycharm/TranVAE/vcf/v1r2_A_30'
    # file_list = get_filelist(path2)
    sample2,truth2 = get_SNP_groundtruth(path2)
    sample = np.concatenate((sample1, sample2), axis=0)
    truth = np.concatenate((truth1, truth2), axis=0)
    path3 = 'D:/Pycharm/TranVAE/vcf/v1r3_A_30'
    sample3, truth3 = get_SNP_groundtruth(path3)
    sample = np.concatenate((sample, sample3), axis=0)
    truth = np.concatenate((truth, truth3), axis=0)
    path3 = 'D:/Pycharm/TranVAE/vcf/v2r1_A_30'
    sample4, truth4 = get_SNP_groundtruth(path3)
    sample = np.concatenate((sample, sample4), axis=0)
    truth = np.concatenate((truth, truth4), axis=0)
    path3 = 'D:/Pycharm/TranVAE/vcf/v2r3_A_30'
    sample5, truth5 = get_SNP_groundtruth(path3)
    sample = np.concatenate((sample, sample5), axis=0)
    truth = np.concatenate((truth, truth5), axis=0)
    path3 = 'D:/Pycharm/TranVAE/vcf/v3r3_A_30'
    sample6, truth6 = get_SNP_groundtruth(path3)
    sample = np.concatenate((sample, sample6), axis=0)
    truth = np.concatenate((truth, truth6), axis=0)
    path3 = 'D:/Pycharm/TranVAE/vcf/v2r1_B_60'
    sample7, truth7 = get_SNP_groundtruth(path3)
    sample = np.concatenate((sample, sample7), axis=0)
    truth = np.concatenate((truth, truth7), axis=0)
    sample = np.concatenate((sample, truth), axis=0)
    sample = sample.astype(np.float32)
    truth = truth.astype(np.float32)
    return sample,truth

def getTransdata(path1):
    file_list = get_filelist(path1)
    sample1 = get_snv_npy(file_list)
    data_shape=sample1.shape
    # input_data=input_data.reshape(6,61440)
    input_data=sample1.reshape(data_shape[2],data_shape[0]*data_shape[1])
    input_data=input_data[-1,:]
    input_data=input_data.reshape(-1,1)
    return input_data

