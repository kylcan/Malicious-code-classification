#### 镜像内容
```
|-- malware_classification_petrichor
    |-- codes # 代码文件
    |-- data # 数据集映射
    |-- train.sh # 执行模型训练过程
    |-- predict.sh  # 执行测试数据预测过程
```
#### 镜像预装包
```angular2
numpy==1.21.2
gensim==4.1.0
tqdm==4.42.1
xgboost==0.90
pandas==1.3.3
joblib==0.14.1
scikit_learn==0.24.2
```
#### 导入方法
由分卷压缩文件解压：
```
cat malware_classification_petrichor.tar.xz* > malware_classification_petrichor.tar.xz
xz -d malware_classification_petrichor.tar.xz
```
导入容器
```shell
docker import - malware_f < malware_classification_petrichor.tar
```
执行训练脚本
```shell
docker run -v /mnt/diskd/malware_classification_bdci/data:/data malware_f sh train.sh
```
执行预测脚本
```shell
docker run -v /mnt/diskd/malware_classification_bdci/data:/data malware_f sh predict.sh
```