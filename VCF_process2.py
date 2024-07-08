import vcf
import numpy as np
import os,re


i=0
# 初始化变量
MAX_COL=2048
UPPER_SIZE=100000
p1=0
p2=0
# 初始化第一个二维矩阵
now_arr1=np.zeros((5,MAX_COL))
now_arr2=np.zeros((6,MAX_COL))
final_arr1=[]
final_arr2=[]
# 当前矩阵起始数据
start_pos1=0
start_pos2=0
# 当前数据在矩阵的位置
arr_pos1=0
arr_pos2=0


def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            if(filename.split('.')[-1]== "vcf"):
                Filelist.append(os.path.join(home, filename))
    return Filelist

def preprocess(path):
    Filelist = get_filelist(path)
    print("文件夹下文件数量：",len(Filelist))
    for file in Filelist:
        p1 = 0
        p2 = 0
        # 初始化第一个二维矩阵
        now_arr1 = np.zeros((5, MAX_COL))
        now_arr2 = np.zeros((6, MAX_COL))
        i=0
        final_arr1 = []
        final_arr2 = []
        # 当前矩阵起始数据
        start_pos1 = 0
        start_pos2 = 0
        # 当前数据在矩阵的位置
        arr_pos1 = 0
        arr_pos2 = 0
        print("文件名",file)
        vcf_reader = vcf.Reader(filename=file)
        for record in vcf_reader:
            a=record.FORMAT.split(":")[1]
            # 判断当前数据是数据一还是数据二
            if a == "DP":
                # print("数据一")
                # 需要创建新矩阵，并将矩阵第一条数据填入
                if p1 == 0:
                    now_arr1 = np.zeros((5, MAX_COL),dtype=np.int32)
                    arr_pos1 = 0
                    # 标志位置1
                    p1 = 1
                    # 得到第一个数据的位置信息
                    start_pos1 = record.POS
                    # print("第一个位置为",start_pos1)
                # 若数据未越界，填入数据
                if arr_pos1<MAX_COL:
                    # print("xiaoyu",record.POS,start_pos1,record.POS-start_pos1-UPPER_SIZE)
                    if record.CHROM == 'X':
                        now_arr1[0][arr_pos1] = 23
                    elif record.CHROM == 'Y':
                        now_arr1[0][arr_pos1] = 24
                    elif record.CHROM == 'M':
                        now_arr1[0][arr_pos1] = 25
                    else:
                        now_arr1[0][arr_pos1] = record.CHROM
                    now_arr1[1][arr_pos1] = record.POS
                    now_arr1[2][arr_pos1] = record.INFO['END']
                    now_arr1[3][arr_pos1] = record.samples[0].data[1]
                    now_arr1[4][arr_pos1] = record.samples[0].data[3]
                    arr_pos1 = arr_pos1 + 1
                else:
                    # print("dayu",record.POS,start_pos1,record.POS-start_pos1-UPPER_SIZE)
                    # 将矩阵加入final_arr1,创建新矩阵
                    final_arr1.append(now_arr1)
                    # print(now_arr1)
                    now_arr1 = np.zeros((5, MAX_COL))
                    arr_pos1 = 0
                    start_pos1 = record.POS
                    if record.CHROM == 'X':
                        now_arr1[0][arr_pos1] = 23
                    elif record.CHROM == 'Y':
                        now_arr1[0][arr_pos1] = 24
                    elif record.CHROM == 'M':
                        now_arr1[0][arr_pos1] = 25
                    else:
                        now_arr1[0][arr_pos1] = record.CHROM
                    now_arr1[1][arr_pos1] = record.POS
                    now_arr1[2][arr_pos1] = record.INFO['END']
                    now_arr1[3][arr_pos1] = record.samples[0].data[1]
                    now_arr1[4][arr_pos1] = record.samples[0].data[3]
                    arr_pos1 = arr_pos1 + 1

            elif a == "AD":
                # print("数据二")
                if p2 == 0:
                    now_arr2 = np.zeros((6, MAX_COL),dtype=np.int32)
                    arr_pos2 = 0
                    # 标志位置1
                    p2 = 1
                    # 得到第一个数据的位置信息
                    start_pos2 = record.POS
                    # print("第一个位置为",start_pos2)
                # 若数据未越界，填入数据
                if arr_pos2<MAX_COL:
                    # print("xiaoyu",record.POS,start_pos2,record.POS-start_pos2-UPPER_SIZE)
                    if record.CHROM == 'X':
                        now_arr2[0][arr_pos2] = 23
                    elif record.CHROM == 'Y':
                        now_arr2[0][arr_pos2] = 24
                    elif record.CHROM == 'M':
                        now_arr2[0][arr_pos2] = 25
                    else:
                        now_arr2[0][arr_pos2] = record.CHROM
                    # print("位置",record.POS)
                    # print(record.samples[0],record.samples[0].data[1])
                    # print(record.samples[0].data[1][0],record.samples[0].data[1][1],record.samples[0].data[1][1])
                    now_arr2[1][arr_pos2] = record.POS
                    now_arr2[2][arr_pos2] = record.samples[0].data[1][0]
                    now_arr2[3][arr_pos2] = record.samples[0].data[1][1]
                    now_arr2[4][arr_pos2] = record.samples[0].data[1][2]
                    now_arr2[5][arr_pos2] = record.samples[0].data[2]
                    arr_pos2 = arr_pos2 + 1
                else:
                    # 将矩阵加入final_arr2,创建新矩阵
                    final_arr2.append(now_arr2)
                    # print(now_arr2)
                    now_arr2 = np.zeros((6, MAX_COL))
                    arr_pos2 = 0
                    start_pos2 = record.POS
                    if record.CHROM == 'X':
                        now_arr2[0][arr_pos2] = 23
                    elif record.CHROM == 'Y':
                        now_arr2[0][arr_pos2] = 24
                    elif record.CHROM == 'M':
                        now_arr2[0][arr_pos2] = 25
                    else:
                        now_arr2[0][arr_pos2] = record.CHROM
                    now_arr2[1][arr_pos2] = record.POS
                    now_arr2[2][arr_pos2] = record.samples[0].data[1][0]
                    now_arr2[3][arr_pos2] = record.samples[0].data[1][1]
                    now_arr2[4][arr_pos2] = record.samples[0].data[1][2]
                    now_arr2[5][arr_pos2] = record.samples[0].data[2]
                    arr_pos2 = arr_pos2 + 1
            i = i+1
            if i%100000==0:
                print(i)
        final_arr1.append(now_arr1)
        final_arr1 = np.array(final_arr1)
        print("尺寸1", final_arr1.shape)
        np.save(file+".CN"+".npy", final_arr1)
        final_arr2.append(now_arr2)
        final_arr2 = np.array(final_arr2)
        print("尺寸2", final_arr2.shape)
        np.save(file+".SNV"+".npy", final_arr2)


#


