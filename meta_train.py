import os
import random
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from meta_test import test
from transformer1 import TransformerClassifier
from Text_CNN import TextCNN
from TransformerTextCNN import TransformerTextCNN
from dataset import DatasetWithTextLabel
from dataloader import EpisodeSampler
from utils import mean_confidence_interval


def seed_everything(seed=42):
    """
    设置整个开发环境的随机种子，保证可复现性
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证cuDNN的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TOKENIZER_PATH = "../bert-base-uncased"
SHOT = 5
TRAIN_WAY = 20
TEST_WAY = 20
TEMPERATURE = 0.01
N_EPISODES = 10
EPOCHS = 40
PRINT_STEP = 500
MAX_LENGTH = 128
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
EMBEDDING_DIM = 512
D_MODEL = 512
NHEAD = 4
NUM_ENCODER_LAYERS = 6
OUTPUT_FILE_PATH = "./log/meta_learning.txt"
CHECKPOINT_PATH = './check_point/pretrain.pth'
EMBEDDING_DICT_PATH = './cwe_desc.pkl'


def prepare_environment():
    seed_everything(SEED)
    print(f"[INFO] Using device: {DEVICE}")

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    return tokenizer
def load_datasets():
    train_dataset = DatasetWithTextLabel(split='train')
    test_dataset = DatasetWithTextLabel(split='test')
    return train_dataset, test_dataset

def build_models(vocab_size, num_classes):
    model_transformer = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_classes=num_classes
    ).to(DEVICE)

    model_textcnn = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        n_filters=N_FILTERS,
        filter_sizes=FILTER_SIZES,
        num_classes=num_classes,
        dropout=0.1
    ).to(DEVICE)

    model_student = TransformerTextCNN(model_transformer, model_textcnn)
    return model_student

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_acc = checkpoint.get('best_acc', 0.0)
    return model, optimizer, best_acc
def load_embedding_dict(path):
    with open(path, 'rb') as f:
        embedding_dict = pickle.load(f)
    return embedding_dict

def train(student, train_loader, optimizer, epoch, tokenizer, embedding_dict):
    student.train()
    losses = 0.0
    accs = 0.0

    with open(OUTPUT_FILE_PATH, 'a') as output_file:
        for idx, episode in enumerate(train_loader):
            codes = episode[0]
            glabels = episode[2]
            labels = torch.arange(TRAIN_WAY).unsqueeze(-1).repeat(1, SHOT + 5).view(-1).to(DEVICE)

            codes_list = [('[CLS]' + str(code)) for code in list(codes)]
            tokenized_codes = tokenizer(
                codes_list,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH,
                return_attention_mask=True
            )
            input_ids = tokenized_codes['input_ids'].to(DEVICE)
            attention_mask = tokenized_codes['attention_mask'].to(DEVICE)

            input_ids = input_ids.view(TRAIN_WAY, SHOT + 5, -1)
            sup, que = input_ids[:, :SHOT].contiguous(), input_ids[:, SHOT:].contiguous()
            sup = sup.view(-1, *sup.shape[2:])
            que = que.view(-1, *que.shape[2:])

            glabels = glabels.view(TRAIN_WAY, SHOT + 5)[:, :SHOT].contiguous().view(-1)
            text_features = torch.stack([torch.Tensor(embedding_dict[cwe_id.item()]) for cwe_id in glabels])
            text_features = text_features.squeeze().to(DEVICE)

            _, sup_im_features = student.forward_with_prompt(sup, text_features)
            sup_im_features = sup_im_features.view(TRAIN_WAY, SHOT, -1).mean(dim=1)  # 平均池化

            _, que_im_features = student(que)

            sim = F.normalize(que_im_features, dim=-1) @ F.normalize(sup_im_features, dim=-1).t()
            loss = F.cross_entropy(sim / TEMPERATURE, labels)

            losses += loss.item()
            _, pred = sim.max(-1)
            accs += labels.eq(pred).sum().float().item() / labels.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % PRINT_STEP == 0 or idx == len(train_loader) - 1:
                log_str = f'Train epoch: {epoch}, step: {idx:3d}, loss: {losses / (idx + 1):.4f}, acc: {accs * 100 / (idx + 1):.2f}'
                output_file.write(log_str + '\n')
                print(log_str)
def main():
    prepare_environment()

    tokenizer = load_tokenizer()
    train_dataset, test_dataset = load_datasets()

    num_classes, train_dataset_labels = train_dataset.calculate_num_classes()
    num_classes_test, test_dataset_labels = test_dataset.calculate_num_classes()
    vocab_size = tokenizer.vocab_size
    episode_sampler = EpisodeSampler(train_dataset_labels, N_EPISODES, TRAIN_WAY, SHOT + 5, False)
    train_loader = DataLoader(train_dataset, batch_sampler=episode_sampler)
    model_student = build_models(vocab_size, num_classes)
    optimizer = optim.SGD(model_student.parameters(), lr=1e-4, momentum=0.9, weight_decay=1e-3)
    model_student, optimizer, best_acc = load_checkpoint(model_student, optimizer, CHECKPOINT_PATH)
    embedding_dict = load_embedding_dict(EMBEDDING_DICT_PATH)
    consecutive_decreases = 0
    best_accuracy = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0

    for epoch in range(EPOCHS):
        train(model_student, train_loader, optimizer, epoch, tokenizer, embedding_dict)

        accuracy_test, f1_test, precision_test, recall_test = test(model_student, test_dataset, tokenizer, embedding_dict, num_classes_test)

        if accuracy_test > best_accuracy:
            best_accuracy = accuracy_test
            best_f1 = f1_test
            best_precision = precision_test
            best_recall = recall_test

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_accuracy
            }, f'./check_point/best_model.pth')

            consecutive_decreases = 0
        else:
            consecutive_decreases += 1

        if consecutive_decreases >= 10:
            print("Accuracy has decreased for 10 consecutive epochs. Stopping training.")
            break

    print(f"Best Test Accuracy: {best_accuracy:.4f}")
    print(f"Best F1-score: {best_f1:.4f}")
    print(f"Best Precision: {best_precision:.4f}")
    print(f"Best Recall: {best_recall:.4f}")

if __name__ == "__main__":
    main()
