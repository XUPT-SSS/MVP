import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import pickle
from tqdm import tqdm

def encode_descriptions(csv_path, output_path, model_name='bert-base-uncased', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = pd.read_csv(csv_path)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    cwe_desc_dict = {}

    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            cwe_id = row['CWE-ID']
            description = str(row['Description'])
            # 分词和编码
            inputs = tokenizer(description, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            # 取 [CLS] token 的最后一层隐藏状态作为句向量
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu()
            cwe_desc_dict[cwe_id] = cls_embedding.numpy()
    with open(output_path, 'wb') as f:
        pickle.dump(cwe_desc_dict, f)

    print(f"Saved encoded CWE description vectors to {output_path}")

if __name__ == "__main__":
    CSV_PATH = 'cwe.csv'         
    OUTPUT_PATH = 'cwe_desc.pkl'  

    encode_descriptions(CSV_PATH, OUTPUT_PATH)
