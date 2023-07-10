
import torch
import numpy as np
import torch.nn as nn



def sattention(embed_dim, num_heads):              
    head_dim = embed_dim // num_heads       
    # 定义查询、键和值的线性层
    query = nn.Linear(embed_dim, embed_dim)
    key = nn.Linear(embed_dim, embed_dim)
    value = nn.Linear(embed_dim, embed_dim)   
    # 定义连接多个头的线性层
    fc = nn.Linear(embed_dim, embed_dim)     
    x = np.load(f"{inter_path}/feature/train_matrix.npy")
    # x = np.load(f"{inter_path}/feature/train_semantic_shaped.npy")
    x = torch.from_numpy(x)
    x = x.to(torch.float32)
    i = 0
    while i < 6:
        batch_size, seq_len, embed_dim = x.shape       
    # 将输入张量分成多个头，并计算查询、键和值
        queries = query(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        keys = key(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        values = value(x).view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)       
    # 计算注意力得分，并缩放得分
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim).float())
        attention_weights = torch.softmax(attention_scores, dim=-1)       
    # 对每个头分别计算注意力上下文向量
        attention_contexts = torch.matmul(attention_weights, values)       
    # 将多个头的注意力上下文向量连接起来，并使用线性层进行变换
        attention_contexts = attention_contexts.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        outputs = fc(attention_contexts)
        print(outputs.shape)
        outputs = outputs.detach().numpy()
        """     
        mean = np.mean(outputs, axis=(1,2), keepdims=True)
        std = np.std(outputs, axis=(1,2), keepdims=True)
        outputs = (outputs - mean) / np.sqrt(std ** 2 + 1e-6)
        """
        
        outputs = outputs / np.linalg.norm(outputs)
        """
        for j in range(0,6):
            outputs[j] = outputs[j] / np.linalg.norm(outputs[j])
            """   
        outputs = torch.from_numpy(outputs)
        outputs = outputs.to(torch.float32)
        outputs = outputs + x
        i = i + 1

    return outputs
    """


def self_attention(x):
    batch_size, seq_len = x.shape
# 计算查询、键和值
    q = nn.Linear(seq_len, seq_len)(x)
    k = nn.Linear(seq_len, seq_len)(x)
    v = nn.Linear(seq_len, seq_len)(x)

# 计算注意力分数
    attention_scores = torch.bmm(q.unsqueeze(1), k.unsqueeze(2))
    attention_scores /= torch.sqrt(torch.tensor(seq_len, dtype=torch.float32))

# 计算注意力权重并加权求和
    attention_weights = nn.functional.softmax(attention_scores, dim=-1)
    attention_output = torch.bmm(attention_weights, v.unsqueeze(2)).squeeze(2)

# 将多头的结果进行拼接
    attention_output = attention_output.view(batch_size, seq_len, -1)

# 通过线性变换将多头自注意力结果转换为模型输出
    output = nn.Linear(seq_len, seq_len)(attention_output)

"""
# create SelfAttention object
inter_path = 'F:/Information_Safety/malware_classification_bdci-master/data/user_data'
