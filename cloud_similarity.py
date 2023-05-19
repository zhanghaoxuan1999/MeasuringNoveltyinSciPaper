#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/11/2 9:57
# @Author : Haoxuan Zhang
# compute cloud similarity
import numpy as np
import math
import pandas as pd


class Model:

    def __init__(self):
        pass

    def cloud_transform(self, y_spor, n):
        """
        x表示云滴， y表示隶属度（这里是钟型隶属度）， 意义是度量倾向的稳定程度；
        Ex表示期望值；En表示熵，He表示超熵通过统计数据样本计算云模型的数字特征
        """

        Ex = y_spor.mean()
        En = abs(y_spor - Ex).mean() * math.sqrt(math.pi / 2)
        He = math.sqrt(y_spor.var() - En ** 2)
        x = np.zeros(1)
        y = np.zeros(1)
        # for q in range(n):
        #     Enn = np.random.randn(1) * He + En
        #     x[q] = np.random.randn() * Enn + Ex
        #     y[q] = np.exp(-(x[q] - Ex) ** 2 / (2 * Enn ** 2))

        return [x, y, Ex, En, He]

    def compute_digital_features(self, data_df, N=1500):
        # 每幅图生成N个云滴
        Y = data_df.to_numpy()
        m = Y.shape[1]
        # Cr = zeros(m,1);
        D = np.zeros((m, 3))
        for i in range(m):
            x, y, Ex, En, He = self.cloud_transform(Y[:, i], N)
            D[i, 0] = Ex
            D[i, 1] = En
            D[i, 2] = He

        return D

    def compute_cloud_sim(self, cloud1, cloud2):
        ex1 = cloud1[0]
        ex2 = cloud2[0]
        en1 = cloud1[1]
        en2 = cloud2[1]
        # 确定云的边界
        cloud1_span = [ex1 - 3 * en1, ex1 + 3 * en1]
        cloud2_span = [ex2 - 3 * en2, ex2 + 3 * en2]
        # print "边界为:",cloud1_span,cloud2_span
        # 确定交界
        if (ex1 + 3 * en1 < ex2 - 3 * en2) or (ex2 + 3 * en2 < ex1 - 3 * en1):
            sim = 0
            return sim
        else:
            cloud_span = [min(ex1 - 3 * en1, ex2 - 3 * en2), max(ex1 + 3 * en1, ex2 + 3 * en2)]
        # 计算出云的交点
        # 规避分母为0的情况
        if en1 == en2:
            x1t = 10e8
        else:
            x1t = (ex2 * en1 - ex1 * en2) / (en1 - en2)
        x2t = (ex1 * en2 - ex2 * en1) / (en1 + en2)
        x1 = min(x1t, x2t)
        x2 = max(x1t, x2t)
        # print "x1:",x1
        # print "x2:",x2
        # 判断交点是否在边界
        alpha = self.computeAcc(ex1 - 3 * en1, ex1, en1)
        # print(ex1, en1)

        # print("look",alpha)
        # 包含关系
        if x1 >= cloud_span[0] and x2 <= cloud_span[1]:
            if cloud1_span[0] <= cloud2_span[0]:
                ol = 2.0 * (cloud2_span[1] - cloud2_span[0]) / (
                        (cloud1_span[1] - cloud1_span[0]) + (cloud2_span[1] - cloud2_span[0]))
            else:
                ol = 2.0 * (cloud1_span[1] - cloud1_span[0]) / (
                        (cloud1_span[1] - cloud1_span[0]) + (cloud2_span[1] - cloud2_span[0]))
            sim = (max(self.computeAcc(x1, ex1, en1), self.computeAcc(x2, ex1, en1)) - alpha) / (1 - alpha) * ol
        else:
            if cloud1_span[0] <= cloud2_span[0]:
                ol = 2.0 * (cloud1_span[1] - cloud2_span[0]) / (
                        (cloud1_span[1] - cloud1_span[0]) + (cloud2_span[1] - cloud2_span[0]))
            else:
                ol = 2.0 * (cloud2_span[1] - cloud1_span[0]) / (
                        (cloud1_span[1] - cloud1_span[0]) + (cloud2_span[1] - cloud2_span[0]))
            if cloud_span[0] <= x1 <= cloud_span[0]:
                miu = self.computeAcc(x1, ex1, en1)
            else:
                miu = self.computeAcc(x2, ex1, en1)
            sim = (miu - alpha) / (1 - alpha) * ol
        return sim

    def computeAcc(self, x, ex, en):
        # print(ex,en)
        a = math.exp(-math.pow((x - ex), 2) / (2 * math.pow(en, 2)))
        # a = math.exp(-math.pow((-3 * en), 2) / (2 * math.pow(en, 2)))
        # a = math.exp(-4.5)

        return a

    def cloud_sim(self, a, b):
        ex1, en1, he1 = a
        ex2, en2, he2 = b
        sim = (ex1 * ex2 + en1 * en2 + he1 * he2) / (
                math.sqrt(ex1 ** 2 + ex2 ** 2) + math.sqrt(en1 ** 2 + en2 ** 2) + math.sqrt(he1 ** 2 + he2 ** 2))
        return sim

    def compute_similarity(self, array):
        m = len(array)

        Sim = np.zeros(m)
        for i in range(m):
            sim = 0
            for j in range(m):
                if i == j:
                    continue
                # 获取第i行的内容
                result = self.compute_cloud_sim(array[i], array[j])
                # result = cloud_sim(dc_array[i], dc_array[j])
                sim += result
            Sim[i] = sim / m

        return Sim


    def compute_innovation(self, sim):
        return 1 - sim


if __name__ == '__main__':
    cloud_model = Model()
    # 文件需要加上  ,Ex,En,He
    digital_features_array = pd.read_csv('threedata/result_problem.csv')

    digital_features_array = np.array(digital_features_array)

    Sim = cloud_model.compute_similarity(digital_features_array)
    Cr = cloud_model.compute_innovation(Sim)
    pd.DataFrame(Cr).to_csv('result/inn_sim/p_innovation.csv')
    pd.DataFrame(Sim).to_csv('result/inn_sim/p_similarity.csv')
