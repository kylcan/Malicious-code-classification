import sys
import os

import numpy as np
import cv2
from utils import load_data




inter_path = 'F:/Information_Safety/malware_classification_bdci-master/data/user_data'
data_path = 'F:/Information_Safety/malware_classification_bdci-master/data/raw_data'
asm_path = f"{data_path}/train/asm"
#pe_path = f"{data_path}/{data_type}/pe"
pe_path = f"{data_path}/train/pe"


#分割{label}/filename文件
# def asmtomatrix(label):

def getMatrixfrom_asm(asm_path,inter_path):
    dirs = asm_path.split('/')
    data_type, file_type = dirs[-2], dirs[-1]
    
    # print(data_type)
    # print(file_type)
    asm_matrix_three = np.zeros(((1,256,256))) # 初始化空数组
    asm_matrix_two = np.zeros((1,1024)) #
    #print(asm_matrix_two.shape)
    i=0
    with open(f"{inter_path}/{data_type}_filename.txt", 'r') as fp:
        
        lines = fp.read().split()
        for line in lines:
            #print(line)
            i=i+1
            print(i)
            f = open(f"F:\\Information_Safety\\malware_classification_bdci-master\\asm_{data_type}_word\\{i}","w")
            f.write(line)
            f.close()
        
            f = open(f"F:\\Information_Safety\\malware_classification_bdci-master\\asm_{data_type}_word\\{i}", 'rb')
            image = np.fromfile(f, dtype=np.ubyte)
            #print(image)
            # print(type(image))
            # print(image.shape)
            image_two =cv2.resize(image, (1,1024)) 
            image_two = image_two.T
            #print(image_two.shape)
            #print(image_two)
            
            image_three = cv2.resize(image,(256,256))
            #rint(image_three)
            
            cv2.imwrite(f"F:\\Information_Safety\\malware_classification_bdci-master\\img_asm_{data_type}\\{i}.png", image_three)
            image_three = image_three.reshape((1,256,256))
        
            
            out = np.linalg.norm(image_two, axis=1)
            #print(image_two)
            image_two = image_two/out
            #print(image_two)
            
            asm_matrix_two = np.concatenate((asm_matrix_two, image_two),axis=0)
            asm_matrix_three = np.concatenate((asm_matrix_three, image_three), axis=0)  # 拼接
        
    X_row=np.size(asm_matrix_three,0)  #计算 X 的行数
    #print(X_row)
    asm_matrix_three = asm_matrix_three[1:X_row,::]
    np.save(f"{inter_path}/feature/{data_type}_asm.npy",asm_matrix_three)
    print(asm_matrix_three.shape)
    
    X_row=np.size(asm_matrix_two,0)  #计算 X 的行数
    #print(X_row)
    asm_matrix_two = asm_matrix_two[1:X_row,::]
    np.save(f"{inter_path}/feature/{data_type}_grey_asm.npy",asm_matrix_two)
    print(asm_matrix_two.shape)
    
    



def getMatrixfrom_pe(pe_path,inter_path):
    dirs = pe_path.split('/')
    data_type, file_type = dirs[-2], dirs[-1]

    pe_matrix_three = np.zeros(((1,256,256))) # 初始化空数组
    pe_matrix_two = np.zeros((1,1024))
    
    #print(pe_matrix_two.shape)

    files = [os.path.join(f"{data_path}/{data_type}/pe",f) for f in os.listdir(f"{data_path}/{data_type}/pe")]
    #print(files)
    # 随机选择3251个文件
    #random_files = random.sample(files, 3251)
    
    i=0
    for full_path in files:
        i=i+1
        print(i)
        
        (filepath, file) = os.path.split(full_path)
        #print(file)
        f = open(f"{data_path}/{data_type}/pe/{file}")
        image = np.fromfile(f, dtype=np.ubyte)
        
        image_two =cv2.resize(image, (1,1024)) 
        image_two = image_two.T
        #print(image_two.shape)
        image_three = cv2.resize(image,(256,256))
        cv2.imwrite(f"F:\\Information_Safety\\malware_classification_bdci-master\\img_pe_{data_type}\\{i}.png", image_three)
        
        image_three = image_three.reshape((1,256,256))
        
        #print(image.shape)
        out = np.linalg.norm(image_two, axis=1)
        #print(image_two)
        image_two = image_two/out
        
        pe_matrix_two = np.concatenate((pe_matrix_two,image_two),axis = 0)
        pe_matrix_three = np.concatenate((pe_matrix_three, image_three), axis=0)  # 拼接
        
    row=np.size(pe_matrix_three,0)  #计算 X 的行数
    #print(row)
    pe_matrix_three = pe_matrix_three[1:row,::]
    print(pe_matrix_three.shape)
    np.save(f"{inter_path}/feature/{data_type}_pe.npy",pe_matrix_three)
    
    row=np.size(pe_matrix_two,0)  #计算 X 的行数
    #print(row)
    pe_matrix_two = pe_matrix_two[1:3252,:]
    print(pe_matrix_two.shape)
    np.save(f"{inter_path}/feature/{data_type}_grey_pe.npy",pe_matrix_two)


getMatrixfrom_asm(asm_path, inter_path)
getMatrixfrom_pe(pe_path,inter_path)
