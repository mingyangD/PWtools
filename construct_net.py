# author: MingYang Da
# time: 2022/2/15
# title: 构建异构网络
import pandas as pd
import numpy as np
from function_store import CHN


def build_net(f_cir_seq, f_cir_GIP, alph, f_pheno_fun, f_pheno_GIP, beta, association, association_change):
    # 构建异构网络
    # 构造circRNA 表型相似性矩阵
    cir_name = list(f_cir_seq.columns)
    fcs = f_cir_seq.values
    fcg = f_cir_GIP.values
    cs = fcs * alph + fcg * (
                1 - alph)  # 有权重wight; fcf[fcf == 0] = fcg[fcf == 0]  # cs = fcf * alpha + fcg * (1 - alpha)
    f_CS = pd.DataFrame(cs, index=cir_name, columns=cir_name)

    pheno_name = list(f_pheno_fun.columns)
    fpf = f_pheno_fun.values
    fpg = f_pheno_GIP.values
    ps = fpf * beta + fpg * (1 - beta)  # 有权重wight  # ps = fpf * beta + fpg * (1 - beta);fpf[fpf == 0] = fpg[fpf == 0]
    f_PS = pd.DataFrame(ps, index=pheno_name, columns=pheno_name)

    # 构建异构网络 all_network
    links1 = f_CS.stack().reset_index()  # circrnas相似性
    links1.columns = ['var1', 'var2', 'value']
    cir_association = links1.loc[(links1['value'] > 0.5) & (links1['var1'] != links1['var2'])]
    links2 = f_PS.stack().reset_index()  # 疾病相似性
    links2.columns = ['var1', 'var2', 'value']
    phe_association = links2.loc[(links2['value'] > 0.5) & (links2['var1'] != links2['var2'])]
    npCombined_Net = np.concatenate((cir_association.values, phe_association.values, association.values,
                                     association_change.values), axis=0)
    npCombined_Net = pd.DataFrame(npCombined_Net)
    all_network = CHN(npCombined_Net)  # 异构网络
    return all_network