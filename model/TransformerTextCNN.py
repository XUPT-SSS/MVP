import torch
import torch.nn as nn
import math
from transformer import TransformerClassifier
from TextCNN import TextCNN
class TransformerTextCNN(nn.Module):
    def __init__(self, transformer_model, textcnn_model):
        super(TransformerTextCNN, self).__init__()
        self.transformer_model = transformer_model
        self.textcnn_model = textcnn_model
        
    def forward(self, x,attention_mask = None):
        # 使用Transformer模型
        _, transformer_features = self.transformer_model(x,attention_mask)        
        # 使用TextCNN模型
        textcnn_output, textcnn_features = self.textcnn_model(transformer_features.permute(1, 0, 2))
        
        return textcnn_output,textcnn_features
    def forward_with_prompt(self,x,semantic_prompt,attention_mask=None):
        _, transformer_features = self.transformer_model.forward_with_prompt(x,semantic_prompt,attention_mask)
        # print(transformer_features.shape)
        textcnn_output, textcnn_features = self.textcnn_model(transformer_features.permute(1, 0, 2))
        # print("textcnn_features.shape",textcnn_features.shape)
        return textcnn_output,textcnn_features
    def forward_with_prompt_attention(self,x,semantic_prompt):
        _, transformer_features = self.transformer_model.forward_with_prompt_attention(x,semantic_prompt)
        # print(transformer_features.shape)
        textcnn_output, textcnn_features = self.textcnn_model(transformer_features.permute(1, 0, 2))
        return textcnn_output,textcnn_features