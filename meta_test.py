import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 128

def test(model, test_dataset, tokenizer, embedding_dict, num_classes_test, test_way=20, shot=5, output_file_path=None):
    model.eval()
    support_set, query_set = [], []
    selected_classes = np.random.choice(num_classes_test, size=test_way, replace=False)
    class_mapping = {original: new for new, original in enumerate(selected_classes)}

    for class_idx in selected_classes:
        class_samples = [sample for sample in test_dataset if sample[1] == class_idx]
        if len(class_samples) < shot:
            continue
        support_set += class_samples[:shot]
        query_set += class_samples[shot:]

    support_features = []
    for i in range(0, len(support_set), shot):
        support_samples = support_set[i:i + shot]
        support_sample_features = []
        for support_sample in support_samples:
            tokenized_support = tokenizer(
                '[CLS] ' + support_sample[0],
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=MAX_LENGTH
            ).to(DEVICE)
            glabels = support_sample[2]
            text_features = torch.Tensor(embedding_dict[glabels]).to(DEVICE)
            with torch.no_grad():
                _, support_feature = model.forward_with_prompt(tokenized_support['input_ids'], text_features)
                support_sample_features.append(support_feature.squeeze())
        support_mean_feature = torch.stack(support_sample_features).mean(dim=0)
        support_features.append(support_mean_feature)

    support_features = torch.stack(support_features)

    query_features = []
    for query_sample in query_set:
        tokenized_query = tokenizer(
            '[CLS]' + query_sample[0],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)
        with torch.no_grad():
            _, query_feature = model(tokenized_query['input_ids'])
            query_features.append(query_feature.squeeze())

    query_features = torch.stack(query_features)

    sim = F.normalize(query_features, dim=-1) @ F.normalize(support_features, dim=-1).t()
    _, pred = sim.max(-1)

    true_labels = [class_mapping[query_sample[1]] for query_sample in query_set]
    pred_labels = [class_mapping[selected_classes[p]] for p in pred.tolist()]

    correct_count = (torch.tensor(pred_labels) == torch.tensor(true_labels)).sum().item()
    accuracy = correct_count / len(query_set)

    precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    if output_file_path:
        with open(output_file_path, 'a') as output_file:
            output_file.write(f"Test Accuracy: {accuracy}\n")

    return accuracy, f1, precision, recall
