#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   features.py
@Contact :   3289218653@qq.com
@License :   (C)Copyright 2021-2022 PowerLZY

@Modify Time      @Author           @Version    @Desciption
------------      -------           --------    -----------
2021-10-04 17:17   PowerLZY&yuan_mes  1.0       程序入口函数
'''

import train
import predict
import feature_engineering
import numpy as np
import utils
import predict
import torch
import torch.nn as nn
import csv
import train



def train_func(inter_path, data_path, number):
	""" Training code."""
	feature_list = ['ember', 'section', 'imports', 'exports', 'words_256', 'semantic',
					'ember_section_ins_words', 'ember_section_ins_semantic']

	feature_engineering.feature_engineering("train",  data_path, inter_path, number)
	train_data_dict = utils.load_data('train', feature_list, inter_path)

	train_lab_path = f"{inter_path}/train_y.npy"
	train_y = np.load(train_lab_path)

	# train.train_model(train_data_dict, train_y, inter_path)


def test_func(inter_path, data_path, number):
	""" Predicting code."""

	# feature_engineering.feature_engineering("test", data_path, inter_path, number)
"""
	print("------------------------ 开始预测 ------------------------")
	# ------------------------ result1 ------------------------
	feature_list1 = ['ember', 'section', 'imports', 'exports']
	result1 = predict.predict_result(feature_list1, inter_path)
	# submit_result(inter_path, result1, "result1")
	# ------------------------ result2 ------------------------
	feature_list2 = ['section', 'exports', 'ember_section_ins_words', 'ember_section_ins_semantic']
	result2 = predict.predict_result(feature_list2, inter_path)
	# submit_result(inter_path, result2, "result2")
	# ------------------------ result2 ------------------------
	feature_list3 = ['section', 'exports', 'words_1000', 'ember_section_ins_semantic']
	result3 = predict.predict_result(feature_list3, inter_path)
	# submit_result(inter_path, result3, "result3")
	# ------------------------ result2 ------------------------
	feature_list4 = ['section', 'exports', 'words_1000', 'semantic']
	result4 = predict.predict_result(feature_list4, inter_path)
	# submit_result(inter_path, result4, "result4")
	print("------------------------ 预测完成 ------------------------")

	# Model ensemble
	result_np = (result1 + result2 + result3 + result4) / 4

	# Submit result
	predict.submit_result(inter_path, result_np, "result")
	print("------------------------ 结果提交 ------------------------")"""


def selffunc(inter_path, data_path, number):
	feature_engineering.feature_engineering("test",  data_path, inter_path, number)
	print(1)
	# data = np.load(f"{inter_path}/feature/test_matrix.npy")
	data = np.load(f"{inter_path}/feature/test_matrix.npy")
	fc = nn.Linear(data.shape[1]*number, 10)
	# fc = train.MyNet()
	fc.load_state_dict(torch.load(f"{inter_path}/fc.pth"))
	arr = []
	with open(f"{inter_path}/test_filename.txt", "r") as f:
    # 读取文件中的每一行，并将其转换为浮点数
		content = [line.strip() for line in f.readlines()]
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			data[i][j] = data[i][j] / np.linalg.norm(data[i][j])
		xi = torch.from_numpy(data[i].reshape(1,data.shape[1]*number))
		xi.requires_grad = True
		y = fc(xi.to(torch.float32))
		y = y.detach().numpy()
		max_index = np.argmax(y)
		arr.append([content[i], max_index])
	with open(f"{inter_path}/self_result.csv", "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(arr)
	



if __name__ == '__main__':

	data_path = 'F:/Information_Safety/malware_classification_bdci-master/data/raw_data'
 
	inter_path = 'F:/Information_Safety/malware_classification_bdci-master/data/user_data'
	number = 1024
	label_path = 'F:/Information_Safety/malware_classification_bdci-master/data/raw_data/train_label.csv'

	train_func(inter_path, data_path, number)
	# train.selftrain(inter_path, number, label_path)
	train.selftrain_level_1(inter_path, number, label_path)

	# test_func(inter_path, data_path, number)
	selffunc(inter_path, data_path, number)

