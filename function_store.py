# author: MingYang Da
# time: 2022/2/14
# title: 储存模型需要的处理数据的函数
# title: 调节整合相似性,各个相似性的权重
# alph:代表circRNAs整合时候的参数; α
# beta:代表phenotype整合时候的参数; β
# mu:代表衰减函数; μ(根据先前文章报道,设置为固定值2.26)
# kap 卡帕:代表步长; k
# lambda :代表整合的异构网络中相关性的阈值; λ(根据先前文章报道,设置为固定值0.5)

import pandas as pd
import math
import numpy as np
from numpy import *
import matplotlib.pylab as plt
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
from scipy import stats
import time
from sklearn.model_selection import LeaveOneOut
import copy
from sklearn.metrics import precision_recall_curve, average_precision_score
import threading


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def DFS2(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    if len(path) == 3:
        return []
    paths = []
    for node in graph[start].keys():
        if node not in path:
            newpaths = DFS2(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def DFS3(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    if len(path) == 4:
        return []
    paths = []
    for node in graph[start].keys():
        if node not in path:
            newpaths = DFS3(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths



def value_sum(graph_path, ap):
    sum = 0
    for ele in ap:
        if len(ele) == 3:
            x = graph_path[ele[0]][ele[1]]
            y = graph_path[ele[1]][ele[2]]
            score_path2 = (x * y) ** 4.52  # 修改了mu  (math.exp(2))  4.52
            sum += score_path2
        elif len(ele) == 4:
            x = graph_path[ele[0]][ele[1]]
            y = graph_path[ele[1]][ele[2]]
            z = graph_path[ele[2]][ele[3]]
            score_path3 = (x * y * z) ** 6.78  # 修改了mu  (math.exp(3))  6.78
            sum += score_path3
    return sum


def CHN(fd):
    graph = {}
    graph_path = {}
    for num_i in range(len(fd)):

        if fd.iloc[num_i, 0] in graph_path:
            continue
        else:
            graph_path[fd.iloc[num_i, 0]] = val(fd, fd.iloc[num_i, 0])
    return graph_path


def val(fd, k):
    generatever = {}
    # for n in range(len(f)):
    #     if f.iloc[n, 0] == k:
    #         generatever[f.iloc[n, 1]] = f.iloc[n, 2]
    # return generatever
    d = fd[fd[0] == k]
    for index_num in range(len(d)):
        generatever[d.iloc[index_num, 1]] = d.iloc[index_num, 2]
    return generatever


def CP(df, insert_positive, label):
    df_label = df[label]
    df = df.drop(label, axis=1)
    df.insert(insert_positive, label, df_label)
    return df


def Normalization(group):
    minVals = group.min()
    maxVals = group.max()
    ranges = maxVals - minVals
    normDataSet1 = (group - minVals) / ranges  # (oldValue-min)  减去最小值
    return normDataSet1


def Standardization(x):
    # :param x: 列表
    # :return: 归一化结果
    nor = list()
    distance = max(x) - min(x)
    for i in x:
        nor.append(round((i - min(x)) / distance, 3))  # 保留3位小数
    return nor


def extract_dataframe_data(neg_df, pos_df, var2, phe_list):
    """
    :param neg_df: 阴性集
    :param pos_df: 阳性集
    :param var2: columns位置
    :param phe_list: phe名字
    :return: 训练集(阴阳 1:1)
    """
    crossv = pos_df
    for phe in phe_list:
        num = len(pos_df[pos_df[var2] == phe])
        f = (neg_df[neg_df[var2] == phe]).sample(n=num, replace=False, random_state=1, axis=0)
        crossv = pd.concat([crossv, f])  # 再检查一下
    return crossv.values


def max_auc_value():
    # 睡觉减寿
    file1 = open('../data/loocv/2/alph_beta', 'r')
    file2 = open('../data/loocv/3/alph_beta', 'r')
    lines1 = file1.readlines()
    lines2 = file2.readlines()
    auc_group1 = []
    auc_group2 = []
    auc_val1 = []
    auc_val2 = []
    for line1, line2 in zip(lines1, lines2):
        hang1 = line1.strip().split('\t')
        hang2 = line2.strip().split('\t')
        auc_group1.append(hang1)
        auc_val1.append(hang1[2])
        auc_group2.append(hang2)
        auc_val2.append(hang2[2])
    max_val1 = max(auc_val1)
    max_val2 = max(auc_val2)
    index_list1 = []
    index_list2 = []
    for list1, list2 in zip(auc_group1, auc_group2):
        if list1[2] == max_val1:
            index_list1.append(list1)
        if list2[2] == max_val2:
            index_list2.append(list2)
    alpha_beta1 = np.matrix(index_list1)[:, 0:2]
    alpha_beta2 = np.matrix(index_list2)[:, 0:2]
    alpha_beta = np.r_[alpha_beta1, alpha_beta2]
    # print(alpha_beta.astype(np.float64).tolist())
    return alpha_beta.astype(np.float64).tolist()


if __name__ == '__main__':
    alpha_beta = max_auc_value()

