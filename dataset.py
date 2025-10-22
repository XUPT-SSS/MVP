import os
import re
import pandas as pd

train_dataset_path = './train.csv' 
test_dataset_path = './test.csv'  

def load_code_dataset(dataset_path):
    dataset = [] 
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error reading CSV file {dataset_path}: {str(e)}")
        return dataset

    # 遍历数据框的每一行
    for index, row in df.iterrows():
        code = row['nor_code']  
        label = row['label'] - 1  
        cwe_id = row['cwe_id']  

        # dataset.append((code, label))
        dataset.append((code, label,cwe_id))
    return dataset
class DatasetWithTextLabel(object):
    def __init__(self,split='test'):

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
        return len(self.dataset)
    def calculate_num_classes(self):
        labels = [label for _,label,_ in self.dataset]
        return len(set(labels)),labels





