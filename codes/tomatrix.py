import numpy as np    

feature = ['words_1000', 'words_250', 'ins_1000', 'ins_250', 'words_300', 'histogram', 'byteentropy', 'strings', 'section', 'semantic', 'imports', 'exports']
# 特征向量文件


def tomatrix(data_type, features, inter_path, number):
    data = np.load(f"{inter_path}/feature/{data_type}_words_{number}.npy")
    # data = np.load(f"{inter_path}/feature/{data_type}_semantic_shaped.npy")
    result = np.empty((data.shape[0], 3, number)) # 初始化为空数组 
    data_array = np.load(f"{inter_path}/feature/{data_type}_words_{number}.npy") 
    result[:, 0, :] = data_array
    data_array = np.load(f"{inter_path}/feature/{data_type}_ins_{number}.npy") 
    result[:, 1, :] = data_array
    data_array = np.load(f"{inter_path}/feature/{data_type}_semantic_length={number}.npy") 
    result[:, 2, :] = data_array
    print(result.shape)    
    np.save(f"{inter_path}/feature/{data_type}_matrix.npy", result)


def integrate(data_type, inter_path, feature, fused_label):
    arr = []
    for f in feature:
        arr.append(np.load(f"{inter_path}/feature/{data_type}_{f}.npy"))
    np.save(f"{inter_path}/feature/{data_type}_{fused_label}.npy", np.hstack(arr).astype(np.float32))


inter_path = 'D:/browserdownload/data/user_data'
data_type = 'train'
