import os
import re
import pandas as pd

train_dataset_path = './train.csv' 
test_dataset_path = './test.csv'  

def load_code_dataset(dataset_path):
    dataset = []  # 初始化一个空列表来存储数据集

    # 使用pandas库读取CSV文件
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error reading CSV file {dataset_path}: {str(e)}")
        return dataset

    # 遍历数据框的每一行
    for index, row in df.iterrows():
        code = row['nor_code']  # 从'code'列获取代码片段
        label = row['label'] - 1  # 从'label'列获取标签
        cwe_id = row['cwe_id']  # 加载cwe_id
        # 将代码片段和对应的标签以元组形式添加到数据集中
        # dataset.append((code, label))
        dataset.append((code, label,cwe_id))
    return dataset
class DatasetWithTextLabel(object):
    def __init__(self,split='test'):
        # 根据数据集名称和拆分（训练、验证或测试）确定数据集路径
        if split == 'train':
            dataset_path = train_dataset_path
        elif split == 'test':
            dataset_path = test_dataset_path
        self.dataset = load_code_dataset(dataset_path)
    def __getitem__(self, i):
        # code, label = self.dataset[i]
        # return code, label
        code,label,cwe_id = self.dataset[i]
        return code,label,cwe_id
    def __len__(self):  
        # 返回数据集的长度（样本数量）
        return len(self.dataset)
    def calculate_num_classes(self):
        # 计算数据集的类别数量
        # label 存储在数据集的第二列
        # labels = [label for _, label in self.dataset]
        labels = [label for _,label,_ in self.dataset]
        return len(set(labels)),labels
# man = DatasetWithTextLabel(split='test')
# print(man.__len__())

# code, label = man[180]
# print(f'code:{code}')
# print(f'label:{label}')



