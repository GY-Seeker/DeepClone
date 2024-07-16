import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# import drow
from BioTransformer.setting import *
import torch.optim as optim # 导入优化器
from BioTransformer.Transformer import Transformer
from BioTransformer.Corpus import TranslationCorpus

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# corpus = TranslationCorpus()
# enc_inputs, dec_inputs, target_batch = corpus.make_Tensor() # 创建训练数据
# del corpus
def train(enc_inputs,dec_inputs,target_batch,save_path):
    model = Transformer() # 创建模型实例
    model = model.cuda()
    criterion = nn.MSELoss() # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # 优化器
    epochs = 100 # 训练轮次
    enc_test = enc_inputs[224:244, :].to(device)
    dec_test = dec_inputs[224:244, :].to(device)
    target_test = target_batch[224:244, :].to(device)
    enc_inputs = enc_inputs[:224,:]
    dec_inputs = dec_inputs[:224,:]
    target_batch = target_batch[:224,:]
    batchs = 64
    model.load()
    test_max = 10000
    for epoch in range(epochs): # 训练 100 轮
        epoch_LOSS = 0
        ii = 0

        for i in range(0,enc_inputs.shape[0],batchs):
            start = time.perf_counter()
            optimizer.zero_grad()  # 梯度清零
            part_enc_inputs = enc_inputs[i:i+batchs,:].to(device)
            part_dec_inputs = dec_inputs[i:i+batchs,:].to(device)
            part_target = target_batch[i:i+batchs,:].to(device)
            outputs, _, _, _ = model(part_enc_inputs, part_dec_inputs) # 获取模型输出
            loss = criterion(outputs.view(-1, tgt_vocab), part_target.view(-1,tgt_vocab))*1024 # 计算损失
            epoch_LOSS += loss
            ii+=1
            loss.backward(retain_graph=True)  # 反向传播
            optimizer.step()  # 更新参数
            end = time.perf_counter()
            runTime = end - start
            print(runTime)
        epoch_LOSS = epoch_LOSS/ii
        optimizer.zero_grad()
        outputs_test, _, _, _ = model(enc_test, dec_test)
        test_loss = criterion(outputs_test.view(-1, tgt_vocab), target_test.view(-1, tgt_vocab)) * 1024  # 计算损失
        if (epoch + 1) % 1 == 0: # 打印损失
            print(f"Epoch: {epoch + 1:04d} cost = {epoch_LOSS:.6f} ")
            print(f"test_cost = {test_loss:.6f}")

        if test_max > test_loss:
            test_max = test_loss
            test = str(test_loss.item())
            model.save(save_path+'/'+test)

def MSE(y, t):
    # 形参t代表训练数据（监督数据）（真实）
    # y代表预测数据
    return 0.5 * np.sum((y - t) ** 2)

def sorted(arr,unique_elements):
    arr = np.round(arr.reshape(-1), 2)
    sorted_indices = np.argsort(unique_elements)
    sorted_elements = unique_elements[sorted_indices]
    for ii in range(len(arr)):
        element = arr[ii]
        d = np.abs(sorted_elements-element)
        index = np.argmin(d)
        arr[ii] = index#sorted_elements[index]
    arr = arr.astype(int)
    return arr

def predict(path1,encode_path,ground_path):
    print(path1)
    corpus = TranslationCorpus(encode_path)
    inputdata, groundTruth = corpus.get_org_file(path1)
    # corpus.make_tensor_2(path1)
    enc_inputs,_, target_batch = corpus.make_Tensor(path1,ground_path)
    # train(enc_inputs,dec_inputs, target_batch)
    model = Transformer() # 创建模型实例
    model = model.cuda()
    batch_size = 64
    pre_data_all = []
    data_truth_all = []
    truth_tests = groundTruth[:, :, 4].reshape(-1)
    # 创建一个大小为 1 的批次，目标语言序列 dec_inputs 在测试阶段，仅包含句子开始符号 <sos>
    for i in range(0, enc_inputs.shape[0], batch_size):
        # start = time.perf_counter()
        enc_test = enc_inputs[i:i+batch_size,:].to(device)
        truth_test = groundTruth[i:i+batch_size, :, 4].reshape(-1)
        target_test = target_batch[i:i + batch_size, :].cpu()
        model.load()
        # model.eval()
        predict, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_test, enc_test) # 用模型进行翻译
        predict = predict.view(-1, tgt_vocab) # 将预测结果维度重塑
        data_pre = corpus.make_SubClone(predict)
        data_pre = data_pre[:, :, 4]
        data_pre = np.abs(data_pre.reshape(-1))
        # plt.plot(data_pre.reshape(-1), 'o', linestyle='', alpha=0.3)
        # data_pre = sorted(data_pre,truth_test)
        if pre_data_all == []:
            pre_data_all = data_pre
        else:
            pre_data_all = np.concatenate((pre_data_all, data_pre), axis=0)
        data_truth = corpus.make_SubClone(target_test)
        data_truth = data_truth[:, :, 4]
        data_truth = np.abs(data_truth.reshape(-1))
        # plt.plot(data_truth.reshape(-1), 'o', linestyle='', alpha=0.3)
        # data_truth = sorted(data_truth, truth_test)

        if data_truth_all == []:
            data_truth_all = data_truth
        else:
            data_truth_all = np.concatenate((data_truth_all, data_truth), axis=0)
    pre = data_truth_all.reshape(-1,1)
    pre_data_all= pre_data_all.reshape(-1,1)
    outdata = np.concatenate((pre, pre_data_all), axis=1)
    classes = sorted(pre_data_all, np.array([0.2,0.3,0.7]))
    classes = classes.reshape(-1,1).astype(int)
    outdata = np.concatenate((outdata, classes), axis=1)
    df  = pd.DataFrame(outdata)
    df.columns=['Normal_CCF','Tumor_CCF','class']
    n_samples = len(df) // 20
    # 随机抽取三分之一的数据
    sampled_df = df.sample(n=n_samples)
    # sampled_df= sampled_df.reset_index()
    # sampled_df['index'] = sampled_df.index

    # sns.set_theme()
    sns.set(font_scale=1.8,style="ticks")
    # plt.figure(figsize=(10, 6), dpi=100)
    ax = sns.jointplot(data=sampled_df, x="Normal_CCF", y="Tumor_CCF",kind="kde")
    ax.plot_joint(sns.scatterplot,alpha=0.3)
    ax.figure.set_size_inches(14, 14)

    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.savefig("./output.png", dpi=300)
    plt.show()
    # drow.matrix(pre_data_all,data_truth_all)


        #data_truth[data_truth<0] = 0/ (np.max(data_truth) - np.min(data_truth))



        # plt.plot(data_truth.reshape(-1),'o',linestyle='',alpha=0.3)
        # plt.show()
        # # data_truth = sorted(groundTruth)
        # end = time.perf_counter()
        # runTime = end - start
        # print(runTime)
        # from sklearn.metrics import f1_score,precision_score,recall_score
        # a = f1_score(data_truth, data_pre, average='weighted')
        # print(a,end=',')
        # a = precision_score(data_truth, data_pre, average='weighted')
        # print(a,end=',')
        # a = recall_score(data_truth, data_pre, average='weighted')
        # print(a)
#
# path1 = 'D:/Pycharm/TranVAE/vcf/v3r3_A_30'
# corpus = TranslationCorpus()
# enc_inputs, dec_inputs, target_batch = corpus.make_Tensor(path1) # 创建训练数据
# del corpus
# train(enc_inputs, dec_inputs, target_batch)
# ax = sns.displot(data_pre.T, fill=True, kind="kde")
# plt.show()
# plt.plot(data_pre.reshape(-1), 'o', linestyle='')
#         # data_pre = sorted(data_pre,truth_test)
#         # data_truth = sorted(data_truth, truth_test)
#         # target_test = target_test.cpu().detach().numpy()
#
#
# #
# # ERR5714649
# path1 = 'D:/Pycharm/TranVAE/vcf/v3r3_A_30_chr1'
# predict(path1)
# path1 = 'D:/Pycharm/TranVAE/vcf/v1r2_A_30'
# predict(path1)
# path1 = 'D:/Pycharm/TranVAE/vcf/v1r3_A_30'
# predict(path1)
# path1 = 'D:/Pycharm/TranVAE/vcf/v2r1_A_30'
# predict(path1)
# path1 = 'D:/Pycharm/TranVAE/vcf/v2r1_B_60'
# predict(path1)
# path1 = 'D:/Pycharm/TranVAE/vcf/v2r3_A_30'
# predict(path1)
# path1 = 'D:/Pycharm/TranVAE/vcf/v3r3_A_30'
# predict(path1)
