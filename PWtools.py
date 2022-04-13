# time: 2022/1/17
# title:
# LOOCV 进行验证
# 交叉验证需要的测试样本和候选样本
# 测试样本: 某表型的关联
# 候选样本: 某表型的未知关联

import numpy as np
import pandas as pd
import copy
# from pwcpa.fun_store import CP, extract_dataframe_data,  value_sum
from function_store import CP, extract_dataframe_data,  value_sum

import construct_net
from multiprocessing import Process, Pool

'''
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




f_pheno_GIP = pd.read_csv('../data/input/pheno_GIPkernel_lable.csv', header=0, index_col=0)
f_pheno_fun = pd.read_csv('../data/input/phe_function_similarity.csv', header=0, index_col=0)
f_CDA = pd.read_csv('../data/input/phe_cir_matrix.txt', sep=',', header=0, index_col=0)
f_cir_GIP = pd.read_csv('../data/input/cir_GIPkernel_lable.csv', header=0, index_col=0)
f_cir_seq = pd.read_csv('../data/input/cir_sequence_similarity.csv', header=0, index_col=0)

# 构造交叉验证的数据
links3 = f_CDA.stack().reset_index()  # 关联
links3.columns = ['var1', 'var2', 'value']
association = links3.loc[links3['value'] == 1]  # 关联集
association_change = CP(association, 0, 'var2')  # 关联集
negative_set = links3.loc[links3['value'] == 0]  # 不关联集
phe_name_list = list(f_pheno_fun.columns)
npCombined_CV = extract_dataframe_data(negative_set, association, 'var2', phe_name_list)  # 包含关联样本和非关联样本 1:1,用于loocv交叉验证

# 将训练集中的阴性数据从全部阴性样本中去除,得到剩余的阴性数据样本  leave_neg_ndarray--ndarray类型
select_neg = npCombined_CV[npCombined_CV[:, 2] == 0]
select_neg_list = select_neg.tolist()
negative_set_list = negative_set.values.tolist()
leave_neg_list = [i for i in negative_set_list if i not in select_neg_list]  # 挑选出剩余的所有阴性样本(总阴-被用作训练集的阴性样本)
leave_neg_ndarray = np.array(leave_neg_list)
# print(leave_neg_ndarray)
# 构建模型进行预测
all_network = construct_net.build_net(f_cir_seq, f_cir_GIP, 0.3, f_pheno_fun, f_pheno_GIP, 0.8, association,
                                      association_change)

def func(phe):
    fp = open(f'./rank/{phe}', 'w')
    temp_list = []
    for one_test in leave_neg_ndarray:
        value1 = one_test[0]
        value2 = one_test[1]
        if value2 == phe:
            allpath = DFS3(all_network, value1, value2, path=[])
            score_allpath = value_sum(all_network, allpath)
            temp_list.append([value1, value2, score_allpath])
    temp_list.sort(key=lambda x: temp_list[2], reverse=True)
    temp_list = temp_list[0:100]
    for l in temp_list:
        hang = '\t'.join(l) + '\n'
        fp.write(hang)
    fp.close()


# process_list = []
# for phe in phe_name_list:
#     print(phe)
#     p = Process(target=func, args=(phe,))
#     p.start()
#     process_list.append(p)
# for i in process_list:
#     p.join()


if __name__=='__main__':
    pool = Pool(3) #创建一个5个进程的进程池

    for phe in phe_name_list:
        print(phe)
        pool.apply_async(func=func, args=(phe,))

    pool.close()
    pool.join()


# for phe in phe_name_list:
#     fp = open(f'./rank/{phe}', 'w')
#     temp_list = []
#     for one_test in leave_neg_ndarray:
#         value1 = one_test[0]
#         value2 = one_test[1]
#         if value2 == phe:
#             allpath = DFS3(all_network, value1, value2, path=[])
#             score_allpath = value_sum(all_network, allpath)
#             item = str(value1)+ '--' + str(value2)
#             temp_list.append([item, score_allpath])
#     temp_list.sort(key=lambda x: x[1], reverse=True)
#     temp_list = temp_list[0:100]
#     for l in temp_list:
#         li = [str(i) for i in l]
#         hang = '\t'.join(li) + '\n'
#         fp.write(hang)
#     fp.close()
'''

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

import argparse
parser = argparse.ArgumentParser(description='predict circRNA phenotype association')
parser.add_argument('-a', "--ass", help='Please enter the associated file with a label')
parser.add_argument('-cs', "--cir_seq", help='Please enter the circRNA sequence similarity file with a label')
parser.add_argument('-cg', "--cir_gip", help='Please enter the circRNA GIP kernel similarity file with a label')
parser.add_argument('-pf', "--phe_fun", help='Please enter the phenotype function similarity file with a label')
parser.add_argument('-pg', "--phe_gip", help='Please enter the phenotype GIP kernel similarity file with a label')
parser.add_argument('-o', "--out_folder", help='Please enter the out folder path; /rank')
parser.add_argument('-t', "--num_threads", help='Please enter the number of num_threads', default=4)


args = parser.parse_args()
association_path = args.ass
cir_seq_sim_path = args.cir_seq
cir_gip_sim_path = args.cir_gip
phe_fun_sim_path = args.phe_fun
phe_gip_sim_path = args.phe_gip
out_path = args.out_folder
num_threads = args.num_threads
num_threads = int(num_threads)

f_pheno_GIP = pd.read_csv(phe_gip_sim_path, header=0, index_col=0)
f_pheno_fun = pd.read_csv(phe_fun_sim_path, header=0, index_col=0)
f_CDA = pd.read_csv(association_path, sep=',', header=0, index_col=0)
f_cir_GIP = pd.read_csv(cir_gip_sim_path, header=0, index_col=0)
f_cir_seq = pd.read_csv(cir_seq_sim_path, header=0, index_col=0)

# 构造交叉验证的数据
links3 = f_CDA.stack().reset_index()  # 关联
links3.columns = ['var1', 'var2', 'value']
association = links3.loc[links3['value'] == 1]  # 关联集
association_change = CP(association, 0, 'var2')  # 关联集
negative_set = links3.loc[links3['value'] == 0]  # 不关联集
phe_name_list = list(f_pheno_fun.columns)
npCombined_CV = extract_dataframe_data(negative_set, association, 'var2', phe_name_list)  # 包含关联样本和非关联样本 1:1,用于loocv交叉验证

# 将训练集中的阴性数据从全部阴性样本中去除,得到剩余的阴性数据样本  leave_neg_ndarray--ndarray类型
select_neg = npCombined_CV[npCombined_CV[:, 2] == 0]
select_neg_list = select_neg.tolist()
negative_set_list = negative_set.values.tolist()
leave_neg_list = [i for i in negative_set_list if i not in select_neg_list]  # 挑选出剩余的所有阴性样本(总阴-被用作训练集的阴性样本)
leave_neg_ndarray = np.array(leave_neg_list)
# print(leave_neg_ndarray)
# 构建模型进行预测
all_network = construct_net.build_net(f_cir_seq, f_cir_GIP, 0.3, f_pheno_fun, f_pheno_GIP, 0.8, association,
                                      association_change)

def func(phe):
    fp = open(out_path + f'/{phe}', 'w')
    temp_list = []
    for one_test in leave_neg_ndarray:
        value1 = one_test[0]
        value2 = one_test[1]
        if value2 == phe:
            allpath = DFS3(all_network, value1, value2, path=[])
            score_allpath = value_sum(all_network, allpath)
            temp_list.append([value1, value2, score_allpath])
    temp_list.sort(key=lambda x: temp_list[2], reverse=True)
    temp_list = temp_list[0:100]
    for l in temp_list:
        hang = '\t'.join(l) + '\n'
        fp.write(hang)
    fp.close()


if __name__=='__main__':
    pool = Pool(num_threads) #创建一个5个进程的进程池

    for phe in phe_name_list:
        print(phe)
        pool.apply_async(func=func, args=(phe,))

    pool.close()
    pool.join()

