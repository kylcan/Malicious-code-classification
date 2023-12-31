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
import pandas as pd
import numpy as np
import joblib
import utils

curPath = os.path.abspath(os.path.dirname("__file__"))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import model
import feature_engineering

warnings.filterwarnings('ignore')


def submit_result(inter_path, result_np, result_name):
    """ Generate the final result file to be submit. """
    result_path = 'D:/browserdownload/data/prediction_result'
    with open(f"{inter_path}/test_filename.txt", 'r') as fp:
        test_filename = fp.read().split()
    result = pd.DataFrame()
    result['filename'] = test_filename
    fam_cols = ['family_' + str(i) for i in range(10)]
    result[fam_cols] = result_np
    result.to_csv(f"{result_path}/{result_name}.csv", index=False)


def predict_result(feature_list, inter_path):
    """ 生成类权重结果 """
    vote_list = []
    test_data_dict = utils.load_data('test', feature_list, inter_path)
    for name, test_data in test_data_dict.items():
        if name in ['words_1000', 'words_300', 'ember_section_ins_words', 'ember_section_ins_semantic']:
            # 使用了TF-IDF的特征做特征选择
            selector = joblib.load(open(f"{inter_path}/models/select_model_{name}.pth", "rb"))
            test_data = selector.transform(test_data)

        clf = model.Model(label=name, inter_path=inter_path)
        y_test = clf.Predict(test_data)
        vote_list.append(y_test)

    labels_loss = pd.read_csv(f"{inter_path}/feature/labels_loss.csv")
    np.save(f"{inter_path}/vote_list_fu", vote_list)
    vote_weight = utils.vote_weight_results(labels_loss, vote_list, feature_list)  # 类权重结果
    # vote = vote_results(vote_list)  # 类平均结果
    return vote_weight

if __name__ == '__main__':

    inter_path = 'D:/browserdownload/test'
    data_path = 'D:/browserdownload/train'

    print("------------------------ 特征工程 ------------------------")
    feature_engineering("test", data_path, inter_path)

    print("------------------------ 开始预测 ------------------------")
    # ------------------------ result1 ------------------------
    feature_list1 = ['ember', 'section', 'imports', 'exports']
    result1 = predict_result(feature_list1, inter_path)
    #submit_result(inter_path, result1, "result1")
    # ------------------------ result2 ------------------------
    feature_list2 = ['section', 'exports', 'ember_section_ins_words', 'ember_section_ins_semantic']
    result2 = predict_result(feature_list2, inter_path)
    #submit_result(inter_path, result2, "result2")
    # ------------------------ result2 ------------------------
    feature_list3 = ['section', 'exports', 'words_1000', 'ember_section_ins_semantic']
    result3 = predict_result(feature_list3, inter_path)
    #submit_result(inter_path, result3, "result3")
    # ------------------------ result2 ------------------------
    feature_list4 = ['section', 'exports', 'words_1000', 'semantic']
    result4 = predict_result(feature_list4, inter_path)
    #submit_result(inter_path, result4, "result4")
    print("------------------------ 预测完成 ------------------------")

    # Model ensemble
    result_np = (result1 + result2 + result3 + result4) / 4

    # Submit result
    submit_result(inter_path, result_np, "result")
    print("------------------------ 结果提交 ------------------------")

