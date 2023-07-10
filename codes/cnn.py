import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


# 定义CNN模型
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool2(self.relu(self.conv2(x)))
        # print(x.shape)
        x = self.pool3(self.relu(self.conv3(x)))
        # print(x.shape)
        x = x.view(-1, 64 * 3 * 3)
        # print(x.shape)
        x = self.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = f"{self.img_dir}/{img_name}.png"
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28,28))
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float()
        label = self.df.iloc[idx, 1]
        label = torch.tensor(label).long()
        return img, label

# 定义超参数
batch_size = 32
num_epochs = 60
learning_rate = 0.001

# 创建数据集和数据加载器
train_dataset = MyDataset(csv_file='F:/Information_Safety/malware_classification_bdci-master/data/raw_data/train_label.csv',img_dir = 'F:\Information_Safety\malware_classification_bdci-master\img\img_pe_train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last= True)
# test_dataset = MyDataset(csv_file='test.csv', root_dir='data/')
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建模型和优化器
model = MyCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
arr = []

# 训练模型
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        # print(inputs.shape)
        outputs = model(inputs)
        # print(outputs.shape)
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # print(1)
        running_loss += loss.item()
        print(i)
        if i % 100 == 99:
            arr.append(loss.item())
            print('[%d, %5d] loss: %.3f' % (epoch+1, num_epochs, running_loss/100))
            running_loss = 0.0
          
#loss图
plt.plot(range(0, 60), arr)
# 添加标题和横纵坐标标签
plt.title('Loss over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
# 显示图表
plt.show()  
            
# 保存模型
torch.save(model.state_dict(), 'cnn_model.pth')


# 设置路径


# 评估模型
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         inputs, labels = data
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))