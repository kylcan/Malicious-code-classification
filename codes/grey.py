import sys
import os
import random

import numpy as np
import cv2

import imageio


from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences

import numpy, scipy, os, array


inter_path = 'F:/Information_Safety/malware_classification_bdci-master/data/user_data'
data_path = 'F:/Information_Safety/malware_classification_bdci-master/data/raw_data'
graph_path = 'F:\Information_Safety\malware_classification_bdci-master\img\img_pe_test'
 
files = [os.path.join(f"{data_path}/test/pe",f) for f in os.listdir(f"{data_path}/test/pe")]
for full_path in files:
    (filepath, file) = os.path.split(full_path)
    filename = f"{data_path}/test/pe/{file}"
    f = open(filename,'rb') # 读入文件
    ln = os.path.getsize(filename) # 文件长度（byte）
    width = 512 # 固定图片宽度为256
    rem = ln%width # 计算余出的字节 
    a = array.array("B") # uint8 数组
    a.fromfile(f,ln-rem) # 将文件读入数组a中，舍去余出的字节
    f.close()            
    g = numpy.reshape(a,(len(a)//width,width)) # 将数组转为二维
    g = numpy.uint8(g)
    graphname = f"{graph_path}/{file}.png"
    imageio.imwrite(graphname,g) # 保存图片



# # 获取源文件夹中所有文件的列表
# files = [os.path.join(f"{data_path}/train/pe",f) for f in os.listdir(f"{data_path}/train/pe")]
# # 随机选择10个文件
# random_files = random.sample(files, 10)
# for full_path in random_files:
#     (filepath, file) = os.path.split(full_path)
#     filename = f"{data_path}/train/pe/{file}"
#     f = open(filename)
#     ln = os.path.getsize(filename) # 文件长度（byte）
#     width = 256 # 固定图片宽度为256
#     rem = ln%width # 计算余出的字节 
#     height = (ln-rem)/width
#     image = np.fromfile(f, dtype=np.ubyte)
#     image = cv2.resize(image,(width,height))
#     graphname = f"graph_path/{file}.png"
#     scipy.misc.imsave(graphname,image) # 保存图片




