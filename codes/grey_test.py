# from PIL import Image

# # 打开原始图像文件
# img = Image.open("F:/Information_Safety/malware_classification_bdci-master/img_pe/0a0f6a7dbb2b077a5433076d13299a08.png")

# # 指定压缩后的图像大小
# size = (512, 512)

# # 将原始图像压缩到指定大小
# img_resized = img.resize(size)

# # 保存压缩后的图像文件
# img_resized.save("F:/Information_Safety/malware_classification_bdci-master/png_test/3.png")

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# 定义CNN模型，这里使用ResNet18作为例子
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # resnet = models.resnet18(pretrained=True)
        resnet = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

# 加载模型
cnn = CNN()

# 打开图像文件并预处理
img = Image.open("F:/Information_Safety/malware_classification_bdci-master/img_pe/0a0f6a7dbb2b077a5433076d13299a08.png").convert('RGB')
print(1)
transform = transforms.Compose([
    transforms.Resize(522),  # 缩放图像到522*522
    transforms.CenterCrop(512),  # 中心裁剪图像到512x512
    transforms.ToTensor(),  # 转换图像到Tensor格式
    # transforms.Lambda(lambda x:x.repeat(3,1,1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化图像
])
print(1)
img_processed = transform(img)
print(1)
# 扩展图像维度并提取特征
img_processed = img_processed.unsqueeze(0)
print(img_processed.shape)
features = cnn(img_processed)

print(features.shape)

# 将静态特征转化为图像
img_reconstructed = transforms.functional.to_pil_image(features.view(-1, 512, 1, 1))

# 保存重建后的图像文件
img_reconstructed.save("F:/Information_Safety/malware_classification_bdci-master/png_test/3.png")