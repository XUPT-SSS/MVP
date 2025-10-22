import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model.transformer import TransformerClassifier
from model.Text_CNN import TextCNN
from model.TransformerTextCNN import TransformerTextCNN
from dataset import DatasetWithTextLabel


# 全局配置与初始化
SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 128
NUM_EPOCHS = 100
PRINT_STEP = 500
OUTPUT_FILE = "./log/pretrain_results.txt"
BEST_MODEL_PATH = "./check_point/pretrain.pth"
PATIENCE = 5  # 训练 loss 连续 PATIENCE 次未下降则 early stop

def seed_everything(seed=SEED):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()
print(f"[INFO] Using device: {DEVICE}")

# 数据加载与准备
tokenizer = AutoTokenizer.from_pretrained("../Codebert-base-uncased")
train_dataset = DatasetWithTextLabel(split='train')
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=128)
num_classes, _ = train_dataset.calculate_num_classes()
vocab_size = tokenizer.vocab_size

# 模型定义
transformer_model = TransformerClassifier(
    vocab_size=vocab_size,
    d_model=512,
    nhead=4,
    num_encoder_layers=6,
    num_classes=num_classes
).to(DEVICE)

textcnn_model = TextCNN(
    vocab_size=vocab_size,
    embedding_dim=512,
    n_filters=100,
    filter_sizes=[3, 4, 5],
    num_classes=num_classes,
    dropout=0.1
).to(DEVICE)

model = TransformerTextCNN(transformer_model, textcnn_model).to(DEVICE)

# 优化器定义
optimizer = optim.SGD(
    model.parameters(),
    lr=5e-4,
    momentum=0.9,
    weight_decay=1e-4
)

# 单轮训练函数
def train(model, dataloader, optimizer, epoch):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    with open(OUTPUT_FILE, 'a') as f:
        for step, (codes, labels) in enumerate(dataloader):
            labels = labels.to(DEVICE)
            codes = ['[CLS] ' + str(code) for code in codes]

            tokens = tokenizer(
                codes,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
                return_attention_mask=True
            )
            input_ids = tokens['input_ids'].to(DEVICE)

            logits, _ = model(input_ids)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = logits.max(dim=1)
            total_acc += (preds == labels).float().mean().item()

            if step % PRINT_STEP == 0 or step == len(dataloader) - 1:
                log_str = (
                    f"[Train] Epoch: {epoch}, Step: {step}, "
                    f"Loss: {total_loss / (step + 1):.4f}, "
                    f"Acc: {total_acc * 100 / (step + 1):.2f}%"
                )
                print(log_str)
                f.write(log_str + '\n')

    return total_loss / len(dataloader)

# 主训练函数
def run():
    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(NUM_EPOCHS):
        avg_loss = train(model, train_dataloader, optimizer, epoch)
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_count = 0

            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
            }, BEST_MODEL_PATH)

            print(f"[INFO] New best loss: {best_loss:.4f}, model saved.")
        else:
            no_improve_count += 1
            print(f"[INFO] No improvement for {no_improve_count} epoch(s)")

        if no_improve_count >= PATIENCE:
            print("[Early Stop] Training loss did not improve for several epochs.")
            break

if __name__ == "__main__":
    run()
