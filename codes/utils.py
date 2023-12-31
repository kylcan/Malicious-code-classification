#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   utils.py
@Contact :   3289218653@qq.com
@License :   (C)Copyright 2021-2022 PowerLZY

@Modify Time      @Author           @Version    @Desciption
------------      -------           --------    -----------
2021-10-04 17:22   PowerLZY&yuan_mes  1.0       归档工具函数
"""
import os
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import log_loss


def file_list(path):
    """返回 filename 列表"""
    list = []
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            list.append(filename)
    return list


def fix_file_index(data_path, inter_path):
    """ Fix the index of training set input and test set result output. """
    train_lab_path = f"{data_path}/train_label.csv"
    train_label = pd.read_csv(train_lab_path)
    train_filename = train_label['filename'].tolist()

    # How to generate current index of test_filename
    # TODO: BE MODIFIED Save
    test_filename = file_list(data_path + '/test/pe/')
    pd.DataFrame({'filename': test_filename}).to_csv(f"{inter_path}/test_filename.txt", header=False, index=False)
    train_filename = train_label['filename']
    train_y = np.array(train_label['family'])
    train_filename.to_csv(f"{inter_path}/train_filename.txt", header=False, index=False)
    np.save(f"{inter_path}/train_y.npy", train_y)
    # 打开 CSV 文件并读取文件名和编号
    csv_file = open(train_lab_path)
    csv_reader = csv.reader(csv_file)
    next(csv_reader)  # 跳过表头
    file_dict = {row[0]: row[1] for row in csv_reader}
    csv_file.close()

    # 打开 TXT 文件并匹配文件名
    arr = []
    with open(f"{inter_path}/train_filename.txt") as txt_file:
        for line in txt_file:
            filename = line.strip()  # 去除字符串中的空格和换行符
            if filename in file_dict:
                arr.append([filename, file_dict[filename]])
    with open(f"{inter_path}/train_filename.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'family'])
        for row in arr:
            writer.writerow(row)


def vote_results(vote_list):
    """ 软投票结果集成 """
    result_ensamle = np.zeros([5024, 10], dtype=float)
    pred_list = []
    final = np.zeros([5024, 10], dtype=float)

    for res in vote_list:
        result_ensamle += res
    for sample in result_ensamle:
        pred_list.append(np.argmax(sample, axis=0))  # 求最大索引

    for i in range(final.shape[0]):
        final[i] = np.zeros([1, 10], dtype=float)
        vote = pred_list[i]
        for res in vote_list:
            if res[i][vote] > final[i][vote]:
                final[i] = res[i]

    return final


def vote_weight_results(labels_loss, vote_list, feature_list):
    """加权软投票结果集成"""
    result_ensamle = np.zeros([5024, 10], dtype=float)
    pred_list = []
    final = np.zeros([5024, 10], dtype=float)
    k = 0
    for res in vote_list:
        weight = np.array(labels_loss[feature_list[k]])
        result_ensamle += res * weight
        k += 1
    for sample in result_ensamle:
        pred_list.append(np.argmax(sample, axis=0))  # 求最大索引

    for i in range(final.shape[0]):
        final[i] = np.zeros([1, 10], dtype=float)
        vote = pred_list[i]
        for res in vote_list:
            if res[i][vote] > final[i][vote]:
                final[i] = res[i]

    return final


def get_class_logloss(x):
    """计算每个类别的logloss"""
    class_pred = np.array(x.iloc[:, :-1])
    class_result = np.array(x['family'])
    class_loss = log_loss(class_result, class_pred, labels=[_ for _ in range(10)])
    return class_loss


def load_data(data_type, feature_list, inter_path):
    """获取train_data_dict, train_labels"""

    train_data_dict = {}
    for feature in feature_list:
        train_data_dict[feature] = np.load(f"{inter_path}/feature/{data_type}_{feature}.npy")

    return train_data_dict

# ------------------------ not used ------------------------
"""
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_t_sne(train_all, train_labels, perplexity = 200):

    X_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(train_all)
    font = {"size": 13,
            "family" : "serif"}
    with plt.style.context("bmh"):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c = plt.cm.Set1(train_labels / 10.), alpha=0.6,
                    cmap=plt.cm.get_cmap('rainbow', 2))
        ax.set_title("Features Visualization", fontdict=font)
        ax.set_ylim([-60, 60])
        ax.set_xlim([-60, 60])


def get_csv_result(path):

    testsub = pd.read_csv(path)
    testsub = testsub.iloc[:, 1:]
    y_test = np.array(testsub)

    return y_test


def get_result_list(y_test, p):

    coutp = {}
    for i in range(len(y_test)):
        indx = -1
        for j in y_test[i]:
            indx+=1
            if j > p:
                if indx not in coutp:
                    coutp[indx] = 0
                coutp[indx]+=1
    return coutp


def get_votew_reslist(train_set, vote_list10, labels_loss3, p):

    pd_tmp = []
    voteN_list = []
    for i in train_set:
        voteN_list.append(vote_list10[i])
        pd_tmp.append(get_result_list(vote_list10[i],p))
    vote_tmp = vote_weight_results(np.array(labels_loss3.iloc[:,train_set]), voteN_list)
    rvote_tmp = get_result_list(vote_tmp,p)
    pd_tmp.append(rvote_tmp)
    return pd_tmp


def how_similar(result1, result2, threshold):

    similar = {}
    similar['total'] = 0
    cout = 0
    for i in range(5024):
        if np.argmax(result1[i], axis=0) == np.argmax(result2[i], axis=0) and result1[i][
            np.argmax(result1[i], axis=0)] > threshold and result2[i][np.argmax(result2[i], axis=0)] > threshold:
            if np.argmax(result1[i], axis=0) not in similar:
                similar[np.argmax(result1[i], axis=0)] = 0
            similar[np.argmax(result1[i], axis=0)] += 1
            cout += 1

    similar['total'] = cout

    return similar


def enhance_family(result1, result2, familys, T):

    y_enhance = result1.copy()

    for family in familys:
        result1_f = result1[:, family]
        result2_f = result2[:, family]
        cout = 0
        for i in range(len(y_enhance)):
            if result1_f[i] > T and result2_f[i] > result1_f[i]:
                y_enhance[i] = result2[i]
                cout += 1
        # print("Family {0} Enhance:{1}!".format(family, cout))

    return y_enhance


def how_different(result1, result2, class_id, threshold1, threshold2):

    dif = {}
    dif['total'] = 0
    dif['logloss_up'] = 0
    dif['logloss_dw'] = 0
    for i in range(5024):
        if np.argmax(result1[i], axis=0) == class_id and result1[i][np.argmax(result1[i], axis=0)] > threshold1 and \
                result1[i][np.argmax(result1[i], axis=0)] < threshold2:
            # if i not in  dif:
            dif[i] = []
            dif[i].append((result1[i][class_id], result1[i][result1[i].argsort()[-2]]))
            dif[i].append((result2[i][class_id], result2[i][result1[i].argsort()[-2]]))
            b = math.log(result1[i][result1[i].argsort()[-1]], 10) - math.log(result2[i][result1[i].argsort()[-1]])
            a = math.log(result1[i][result1[i].argsort()[-2]], 10) - math.log(result2[i][result1[i].argsort()[-2]])
            dif[i].append(b)
            dif[i].append(a)
            dif['total'] += 1
            dif['logloss_up'] += b
            dif['logloss_dw'] += a

    return dif

"""