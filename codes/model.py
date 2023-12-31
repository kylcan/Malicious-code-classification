#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   features.py
@Contact :   3289218653@qq.com
@License :   (C)Copyright 2021-2022 PowerLZY

@Modify Time      @Author           @Version    @Desciption
------------      -------           --------    -----------
2021-10-04 17:17   PowerLZY&yuan_mes  1.0       自定义模型
'''

import joblib
import numpy as np
import pandas as pd
import math
from utils import get_class_logloss
from sklearn.model_selection import StratifiedKFold, cross_val_score


class Model(object):
    """ Model training, cross-validation and predicting. """

    def __init__(self, model=None, X=None, y=None, label=None, inter_path=None, labels_loss = None):
        self.model = model
        self.X = X
        self.y = y
        self.label = label
        self.inter_path = inter_path
        self.labels_loss = labels_loss

    def CrossValidation(self, n_splits):
        cv = StratifiedKFold(n_splits=n_splits)
        cv_scores = cross_val_score(self.model, self.X, self.y, scoring='neg_log_loss', cv=cv)
        print(f"Logloss_mean after {n_splits} folds: {-np.mean(cv_scores):.6f}")

    def get_class_weight(self):
        # 计算每个类别在训练集上的logloss
        y_train = self.model.predict_proba(self.X)
        kclass_logloss = pd.DataFrame(y_train)
        kclass_logloss['family'] = self.y
        kclassloss = kclass_logloss.groupby('family').apply(get_class_logloss)

        self.labels_loss[self.label] = kclassloss
        self.labels_loss[np.isnan(self.labels_loss)] = 2  # logx 大于0 即可
        self.labels_loss = self.labels_loss.astype(float)
        self.labels_loss[self.label] = - self.labels_loss[self.label].apply(math.log)
        self.labels_loss[self.labels_loss < 0] = 0
        print(f"--------------------- {self.label}训练完成！ ---------------------")

        return self.labels_loss[self.label]

    def Fit(self):
        self.model.fit(self.X, self.y)
        joblib.dump(self.model, f"{self.inter_path}/models/XGB_model_{self.label}.pkl")

    def Predict(self, test_X):
        model = joblib.load(f"{self.inter_path}/models/XGB_model_{self.label}.pkl")
        test_y = model.predict_proba(test_X)
        return test_y
