# -*- coding: utf-8 -*-
import math
import numpy as np
from numpy import linalg as la
from scipy.spatial.distance import pdist
import os
import sys
import scipy.io as scio

class AILE:
    def __init__(self, epsilon=1e-10, maxstep=500):
        self.epsilon = epsilon
        self.maxstep = maxstep
        self.T = None  # 未标记点间的转换矩阵
        self.Y = None  # 标签数矩阵
        self.Y_clamp = None  # 已知标签数据点的标签矩阵
        self.labels = None

    def init_param(self, y_data):
        self.Y = y_data
        self.Y_clamp = self.Y
        self.T = self.cal_tran_mat(y_data)  # n*n
        return

    def cal_tran_mat(self, data):
        # 计算转换矩阵, 即构建图
        numInstances = data.shape[0]
        numLabels = data.shape[1]
        p_matrix = np.zeros((numInstances, numInstances))
        new_p_matrix = np.zeros((numInstances, numInstances))
        instanceTolabel = np.zeros((numInstances,numLabels))
        labelToinstance = np.zeros((numLabels,numInstances))
        count_instance = np.zeros(numLabels)
        count_label = np.zeros(numInstances)
        for i in range(numInstances):
            for j in range(numLabels):
                if data[i, j] != 0 :
                    count_label[i] += 1
                    count_instance[j] += 1

        for i in range(numInstances):
            for j in range(numLabels):
                if data[i][j] != 0:
                    if count_label[i] != 0:
                        instanceTolabel[i, j] = 1.0 / count_label[i]
                    if count_instance[j] != 0:
                        labelToinstance[j, i] = 1.0 / count_instance[j]

        for i in range(numInstances):
            for j in range(numInstances):
                weight = 0
                for k in range(numLabels):
                    weight += instanceTolabel[i, k] * labelToinstance[k, j]
                p_matrix[i,j] = weight

        for i in range(numInstances):
            for j in range(numInstances):
                new_p_matrix[i, j] = p_matrix[j,i]

        return new_p_matrix

    def fit(self, y_data,alpha):
        # 训练主函数
        self.init_param(y_data)
        step = 0
        while step < self.maxstep:
            step += 1
            new_Y = self.T @ self.Y  # 更新标签矩阵
            new_Y = alpha * new_Y +(1-alpha) * self.Y_clamp  # clamp
            if np.abs(new_Y - self.Y).sum() < self.epsilon:
                break
            self.Y = new_Y
        print(step)
        self.labels = self.Y

        for i in range(len(self.labels)):
            self.labels[i] = np.exp(self.labels[i]) / np.exp(self.labels[i]).sum()
        return

def chebyshev(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.max(np.abs(distribution_real-distribution_predict), 1)) / height

def cosine(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sum(distribution_real*distribution_predict, 1) / (np.sqrt(np.sum(distribution_real**2, 1)) * \
                                                                       np.sqrt(np.sum(distribution_predict**2, 1)))) / height
def clark(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.sqrt(np.sum((distribution_real-distribution_predict)**2 / (distribution_real+distribution_predict)**2, 1))) / height

def canberra(distribution_real, distribution_predict):
    height = distribution_real.shape[0]
    return np.sum(np.abs(distribution_real-distribution_predict) / (distribution_real+distribution_predict)) / height

def intersection(distribution_real, distribution_predict):
    height, width = distribution_real.shape
    inter = 0.
    for i in range(height):
        for j in range(width):
            inter += np.min([distribution_real[i][j], distribution_predict[i][j]])
    return inter / height

def dTol(y, threshold = 0.5):
    label_shape = y.shape
    labels = np.zeros(label_shape)
    for i in range(label_shape[0]):
        threshold_tmp = 0
        index = np.argsort(y[i])
        for j in range(len(index)):
            if threshold_tmp >= threshold:
                break
            threshold_tmp += y[i][index[len(index)-j-1]]
            labels[i][index[len(index)-j-1]] = 1

    return labels

if __name__ == '__main__':

    datesetName = []
    alphaValue = []
    # datesetName.append("Artificial")
    datesetName.append("SJAFFE")
    # datesetName.append("SBU_3DFE")
    # datesetName.append("Yeast_spo5")
    # datesetName.append("Yeast_dtt")
    # datesetName.append("Yeast_cold")
    # datesetName.append("Yeast_heat")
    # datesetName.append("Yeast_spo")
    # datesetName.append("Yeast_diau")
    # datesetName.append("Yeast_elu")
    # datesetName.append("Yeast_cdc")
    # datesetName.append("Yeast_alpha")
    # datesetName.append("Movie")

    # alphaValue.append(0.5)
    alphaValue.append(0.5)
    # alphaValue.append(0.5)
    # alphaValue.append(0.7)
    # alphaValue.append(0.9)
    # alphaValue.append(0.8)
    # alphaValue.append(0.8)
    # alphaValue.append(0.8)
    # alphaValue.append(0.8)
    # alphaValue.append(0.9)
    # alphaValue.append(0.9)
    # alphaValue.append(0.9)
    # alphaValue.append(0.3)

    for i in range(len(datesetName)):
        datafile = "LDL DataSets\\" + datesetName[i] + ".mat"
        temp_alpha = alphaValue[i]
        print(temp_alpha)
        data = scio.loadmat(datafile)
        # label_shape = data['labels'].shape
        mll_labels = dTol(data['labels'])
        LPA = AILE(maxstep=200)
        LPA.fit(mll_labels,temp_alpha)
        pre_labels = LPA.labels

        result_str0 = str("chebyshev：" + datesetName[i] + "：" + str(chebyshev(data['labels'], pre_labels)))
        print(result_str0)

        result_str1 = str("clark：" + datesetName[i] + "：" + str(clark(data['labels'], pre_labels)))
        print(result_str1)

        result_str2 = str("canberra：" + datesetName[i] + "：" + str(canberra(data['labels'], pre_labels)))
        print(result_str2)

        result_str3 = str("Cosine：" + datesetName[i] + "：" + str(cosine(data['labels'], pre_labels)))
        print(result_str3)

        result_str4 = str("intersection：" + datesetName[i] + "：" + str(intersection(data['labels'], pre_labels)))
        print(result_str4)

        print('\n')










