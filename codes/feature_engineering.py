#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   feature_engineering.py
@Contact :   3289218653@qq.com
@License :   (C)Copyright 2021-2022 PowerLZY

@Modify Time      @Author           @Version    @Desciption
------------      -------           --------    -----------
2021-10-04 17:17   PowerLZY&yuan_mes  1.0         None
"""
import sys
import os

from tqdm import tqdm
import joblib
import features
import utils
import pandas as pd
import numpy as np
import tomatrix
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn as nn
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import PathLineSentences
import selfattention


def feature_engine(obj, sample_path, inter_path):
    """ Feature engineering for different features. """
    dirs = sample_path.split('/')
    # Data belongs to 'train' or 'test' and file belongs to 'pe' or 'asm'
    data_type, file_type = dirs[-2], dirs[-1]
    with open(f"{inter_path}/{data_type}_filename.txt", 'r') as fp:
        filename = fp.read().split()
    arr = np.zeros((len(filename), obj.dim))

    if file_type == 'pe':
        with tqdm(total=len(filename), ncols=80, desc=f"{data_type}_{obj.name}") as pbar:
            for i, sample in enumerate(filename):
                with open(f"{sample_path}/{sample}", "rb") as f:
                    bytez = f.read()
                arr[i, :] = obj.feature_vector(bytez)
                pbar.update(1)
    else:  # file_type == 'asm'
        with tqdm(total=len(filename), ncols=80, desc=f"{data_type}_{obj.name}") as pbar:
            for i, sample in enumerate(filename):
                with open(f"{sample_path}/{sample}.asm", "rb") as f:
                    stringz = f.read().decode('utf-8', errors='ignore')
                arr[i, :] = obj.feature_vector(stringz)
                pbar.update(1)
    np.save(f"{inter_path}/feature/{data_type}_{obj.name}.npy", arr)


def feature_tfidf_df(obj, sample_path, inter_path):
    """ Save the words of all samples to a DataFrame fot tf-idf input. """
    dirs = sample_path.split('/')
    data_type, file_type = dirs[-2], dirs[-1]
    with open(f"{inter_path}/{data_type}_filename.txt", 'r') as fp:
        filename = fp.read().split()
    if file_type == 'asm':
        filename = [f + '.asm' for f in filename]
    all_word_feature = []
    with tqdm(total=len(filename), ncols=80, desc=f"{obj.name_tfidf}_{data_type}") as pbar:
        for i, sample in enumerate(filename):
            with open(f"{sample_path}/{sample}", "r", encoding='utf-8', errors='ignore') as f:
                all_word_feature.append(obj.tfidf_features(f))
            pbar.update(1)
    word_feature = pd.DataFrame({'filename':filename, "word_feature": all_word_feature})
    word_feature.to_csv(f"{inter_path}/feature/{data_type}_{obj.name_tfidf}_tfidf.csv", index=False)

    # return word_feature


def model_tfidf(obj, sample_path, inter_path, tfidf_params):
    """ Save the tf-idf model """
    train_words_ = pd.read_csv(f"{inter_path}/feature/train_{obj.name_tfidf}_tfidf.csv")
    delect789 = list(np.load(f"{inter_path}/train_filename_de.npy")) # 不要训练集的789 太少了
    train_words_de = train_words_.loc[~train_words_.filename.isin(delect789)]

    test_words_ = pd.read_csv(f"{inter_path}/feature/test_{obj.name_tfidf}_tfidf.csv")
    all_words_ = train_words_de.append(test_words_)

    vectorizer = TfidfVectorizer(**tfidf_params)
    vectorizer.fit(all_words_.word_feature.tolist())
    joblib.dump(vectorizer, open(f"{inter_path}/models/TFIDF_model_{obj.name_tfidf}_{tfidf_params['max_features']}.pth", "wb"))


def feature_tfidf_np(data_type, name_tfidf, inter_path, max_features):
    """ Save the tf-idf feature with numpy """
    vectorizer = joblib.load(open(f"{inter_path}/models/TFIDF_model_{name_tfidf}_{max_features}.pth", "rb"))
    vectorizer.max_features = max_features
    words_ = pd.read_csv(f"{inter_path}/feature/{data_type}_{name_tfidf}_tfidf.csv")
    words = vectorizer.transform(words_.word_feature.tolist())
    np.save(f"{inter_path}/feature/{data_type}_{name_tfidf}_{max_features}.npy", words.toarray())


def feature_asm2txt(sample_path, inter_path):
    """ Save the opcode of all samples to a txt for asm2vec input. """
    dirs = sample_path.split('/')
    data_type, file_type = dirs[-2], dirs[-1]

    def asm2txt_by_datatype(spath, dtype):
        with open(f"{inter_path}/{dtype}_filename.txt", 'r') as fp:
            filenames = fp.read().split()

        if dtype == 'test':
            spath = spath.replace('train', 'test')

        with tqdm(total=len(filenames), ncols=80, desc=f"{dtype}_asm2txt") as pbar:
            for filename in filenames:
                with open(os.path.join(spath, filename) + '.asm', "r", encoding='utf-8', errors='ignore') as fp:
                    opline_list = features.OpcodeInfo().asm_to_txt(fp)
                f = open(os.path.join(f"{inter_path}/semantic/", filename) + '.txt', 'w+', encoding = 'utf-8')
                for line in opline_list:
                    f.write(line)
                    f.write('\n')
                f.close()
                pbar.update(1)

    asm2txt_by_datatype(sample_path, data_type)
    asm2txt_by_datatype(sample_path, 'test')


def feature_asm2vec(data_type, inter_path):
    """Feature engineering for asm2vec feature."""

    if data_type == "train":
        # TODO : 模型空判断
        # Train a Word2vec model by mixing traing set and test set
        print("------------------------ 训练asm2vec模型 ------------------------")
        sentences = PathLineSentences(f"{inter_path}/semantic/")
        model = Word2Vec(sentences=sentences, vector_size= 1024, window=5, min_count=5, workers=5)
        model.wv.save_word2vec_format(f"{inter_path}/models/asm2vec.bin", binary=True, sort_attr='count')

    # Load the trained Word2vec model
    model_wv = KeyedVectors.load_word2vec_format(f"{inter_path}/models/asm2vec.bin", binary=True)

    print("------------------------ 生成asm2vec特征 ------------------------")
    with open(f"{inter_path}/{data_type}_filename.txt", 'r') as fp:
        filename = fp.read().split()
    # Feature engineering for generating string vector features
    obj =features.StringVector()
    arr = np.zeros((len(filename), obj.dim))
    with tqdm(total=len(filename), ncols=80, desc=obj.name) as pbar:
        for i, file in enumerate(filename):
            with open(f"{inter_path}/semantic/{file}.txt", "rb") as f:
                stringz = f.read().decode('utf-8', errors='ignore')
            lines = ' '.join(stringz.split('\n'))
            raw_words = list(set(lines.split()))
            arr[i, :] = obj.feature_vector((model_wv, raw_words))
            pbar.update(1)
    arr[np.isnan(arr)] = 0
    np.save(f"{inter_path}/feature/{data_type}_semantic.npy", arr)


def feature_fusion(data_type, fused_label, features, inter_path):

    arr = []
    for f in features:
        arr.append(np.load(f"{inter_path}/feature/{data_type}_{f}.npy"))
    np.save(f"{inter_path}/feature/{data_type}_{fused_label}.npy", np.hstack(arr).astype(np.float32))


def fea_shape(data_type, feature, inter_path, number):
    arr = np.load(f"{inter_path}/feature/{data_type}_{feature}.npy")
    if arr.shape[1] >= number:
        cropped_matrix =arr[:, 0:number]
        np.save(f"{inter_path}/feature/{data_type}_{feature}_length={number}.npy", cropped_matrix)
    else:
        cropped_matrix = np.zeros((arr.shape[0], int(number)))
        cropped_matrix[:, 0:arr.shape[1]] = arr[:,0:arr.shape[1]]
        np.save(f"{inter_path}/feature/{data_type}_{feature}_length={number}.npy", cropped_matrix)


def attention1(data_type, inter_path, number):
    result = selfattention.sattention(embed_dim = number, num_heads = 8)
    np.save(f"{inter_path}/feature/{data_type}_self_sttention.npy", result.detach().numpy())


def fea_semantic(data_type, inter_path, number):
    arr = np.load(f"{inter_path}/feature/{data_type}_semantic.npy")
    arrr = list()
    embedding = nn.Embedding(6000, number)
    for i in range(0, arr.shape[0]):
        input = torch.LongTensor(arr[i])
        input = embedding(input[0])
        arrr.append(input.detach().numpy())
    arrr = np.array(arrr)
    print(arrr.shape)
    np.save(f"{inter_path}/feature/{data_type}_semantic_length={number}.npy", arrr)


def feature_engineering(data_type, data_path, inter_path, number):

    pe_path = f"{data_path}/{data_type}/pe"
    asm_path = f"{data_path}/{data_type}/asm"
    
    # if data_type == 'train':
    #     print("------------------------ 生成索引文件 ------------------------")
    #     # TODO：file_index without under sample
    #     # utils.fix_file_index(data_path, inter_path)
        
    #     print("------------------------ 生成TF-IDF词库 ------------------------")
    #     # feature_tfidf_df(features.StringExtractor(), pe_path, inter_path)
    #     # feature_tfidf_df(features.StringExtractor(), pe_path.replace('train', 'test'), inter_path)

    #     # feature_tfidf_df(features.OpcodeInfo(), asm_path, inter_path)
    #     # feature_tfidf_df(features.OpcodeInfo(), asm_path.replace('train', 'test'), inter_path)
        
        
    #     print("------------------------ 生成TF-IDF模型 ------------------------")
    #     words_tf_params2 = {'max_features': number}
    #     model_tfidf(features.StringExtractor(), pe_path, inter_path, words_tf_params2)
    #     ins_tf_params = {'stop_words' : [';'], 'ngram_range' : (1, 3), 'max_features': number}
    #     model_tfidf(features.OpcodeInfo(), asm_path, inter_path, ins_tf_params)

    #     print("------------------------ 生成asm2vec词库 ------------------------")
    #     # feature_asm2txt(asm_path, inter_path)

    
    # print("------------------------ 生成TF-IDF特征 ------------------------")
    # feature_tfidf_np(data_type, 'words', inter_path, max_features=number)
    # feature_tfidf_np(data_type, 'ins', inter_path, max_features=number)

    # print("------------------------ 生成ember特征 ------------------------")
    # # pe_objs = [features.ByteHistogram(), features.ByteEntropyHistogram(), features.StringExtractor()]
    # # for obj in pe_objs:
    # #     feature_engine(obj, pe_path, inter_path)

    # # asm_objs = [features.SectionInfo(), features.ImportsInfo(), features.ExportsInfo()]
    # # for obj in asm_objs:
    # #     feature_engine(obj, asm_path, inter_path)
    
    # # ------------------------ 生成asm2vec特征 ------------------------
    # # feature_asm2vec(data_type, inter_path)
    # """
    # """
    # print("--------------------------特征融合-------------------------------------")
    # # fea_shape(data_type, 'histogram', inter_path, number)    
    # feature = [f"words_{number}", f"ins_{number}","semantic"]
    # fea_semantic(data_type, inter_path, number)
    # tomatrix.tomatrix(data_type, feature, inter_path, number)
    
    if data_type == "train":
        print("----------------------自注意力-----------------------------------------")
        attention1(data_type, inter_path, number)
        
    
