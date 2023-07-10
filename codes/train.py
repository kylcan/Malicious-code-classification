#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Contact :   3289218653@qq.com
@License :   (C)Copyright 2021-2022 PowerLZY

@Modify Time      @Author               @Version    @Desciption
------------      -------               --------    -----------
2021-10-04 15:03   PowerLZY&yuan_mes       1.0      保存训练模型，特征选择模型和单特征类别权重
'''
import warnings
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import class_weight
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from utils import load_data
from feature_engineering import feature_engineering
from model import Model
import csv
import matplotlib.pyplot as plt
import torch.optim as optim
curPath = os.path.abspath(os.path.dirname("__file__"))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

warnings.filterwarnings('ignore')


def train_model(train_data_dict, train_labels, inter_path):

    model_A = XGBClassifier(
        objective='multi:softprob',
        num_class=10,
        max_depth=6,
        n_estimators=90,
        learning_rate=0.1,
        eval_metric='mlogloss',
        use_label_encoder=False
    )

    classes_weights = class_weight.compute_sample_weight(
        class_weight='balanced',
        y=train_labels
    )

    labels_loss = pd.DataFrame()
    print(f"------------------------ 开始训练 ------------------------")
    for name, train_data in train_data_dict.items():

        if name in ['words_256', 'ember_section_ins_words', 'ember_section_ins_semantic']:
            # 使用了TF-IDF的特征做特征选择
            selector = SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=200)).fit(train_data, train_labels,
                                                                                        sample_weight=classes_weights)

            joblib.dump(selector, open(f"{inter_path}/models/select_model_{name}.pth", "wb"))
            train_data = selector.transform(train_data)

        clf = Model(model_A, train_data, train_labels, name, inter_path, labels_loss)
        clf.Fit()

        labels_loss[name] = clf.get_class_weight()

    labels_loss[np.isnan(labels_loss)] = 0
    labels_loss[labels_loss < 0] = 0
    labels_loss.to_csv(f"{inter_path}/feature/labels_loss.csv", index=False)
    print(f"------------------------ 训练完成 ------------------------")


def selftrain_level_1(inter_path ,number, label_path):
    x = np.load(f"{inter_path}/feature/train_self_sttention.npy")
    # 定义输入数据和目标值
    arr = []
    with open(f"{inter_path}/train_filename.csv", mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            arr.append(row)
    # print(type(arr[0]['family']))
    one_hot_encoding = np.eye(10)
    one_hot_encoding = torch.from_numpy(one_hot_encoding)
    # 定义全连接层
    linear = nn.Linear(in_features = x.shape[1]*number, out_features = 10)

    # 定义损失函数
    loss_fn = nn.MSELoss()
    # 定义优化器
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)
    arr2 = []
    for i in range(x.shape[0]):
    # 前向传播
        for j in range(x.shape[1]):
            x[i][j] = x[i][j] / np.linalg.norm(x[i][j])
        xi = torch.from_numpy(x[i].reshape(1, x.shape[1]*number))
        xi.requires_grad = True
        # print(xi.requires_grad)
        # print(xi.shape)
        xi = xi.to(torch.float32)
        y_pred = linear(xi)
        y = one_hot_encoding[int(arr[i]['family'])]
    # 计算损失函数
        y = y.to(torch.float32)
        y_pred = y_pred.to(torch.float32)
        loss = loss_fn(y_pred, y)
        loss = loss.to(torch.float32)
        # print(y_pred)
    # 反向传播
        loss.backward()
    # 更新模型参数optimizer调用 `step()` 方法
        optimizer.step()
    # 清空梯度
        optimizer.zero_grad()
        if i % 100 == 99:
            arr2.append(loss.item())
    # 打印损失函数
            print('Epoch [{}/{}], Loss: {:.4f}'.format(i+1, 1, loss.item()))
# 创建一个折线图
    plt.plot(range(0, 58), arr2)
# 添加标题和横纵坐标标签
    plt.title('Loss over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
# 显示图表
    plt.show()
    torch.save(linear.state_dict(), f"{inter_path}/fc.pth")




if __name__ == '__main__':

    data_path = 'F:/Information_Safety/malware_classification_bdci-master/data/raw_data'
    inter_path = 'F:/Information_Safety/malware_classification_bdci-master/data/user_data'
    label_path = 'F:/Information_Safety/malware_classification_bdci-master/data/raw_data/train_label.csv'
    number = 1024
    feature_list = ['ember', 'section', 'imports', 'exports', 'words_256', 'semantic', 'ember_section_ins_words', 'ember_section_ins_semantic']

    print(f"------------------------ 训练集特征工程 ------------------------")
    feature_engineering("train", data_path, inter_path, number)
    train_data_dict = load_data('train', feature_list, inter_path)
    # label(label_path)
    selftrain_level_1(inter_path, number, label_path)
    # train_lab_path = f"{inter_path}/train_y.npy"
    # train_y = np.load(train_lab_path)

    # train_model(train_data_dict, train_y, inter_path)
