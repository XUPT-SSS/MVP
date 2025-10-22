import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        max_len = max(x.size(0), self.max_len)
        pe = torch.zeros(max_len, self.d_model, device=x.device)
        position = torch.arange(0, max_len, dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model)).to(x.device)
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        return x + pe[:x.size(0), :]
    

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead) for _ in range(num_encoder_layers)
        ])
        self.fc = nn.Linear(d_model, num_classes)
        self.semantic_prompt_projection = nn.Linear(768, d_model)
        # 添加MLP层
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(),
            nn.Linear(512, 512)    
        )
        # self.attention_layer = nn.MultiheadAttention(d_model, nhead)
    def forward(self, x, attention_mask=None):
        x = self.embedding(x)
        x += self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()
        
        for layer in self.transformer_encoder_layers:
            x = layer(x, src_key_padding_mask=attention_mask)
        x2 = x
        x = x[0] 
        x = self.fc(x)
        return x,x2

    def forward_with_prompt(self, x, semantic_prompt,attention_mask=None):
        x = self.embedding(x)
        x += self.pos_encoder(x)
        # print(x.shape)
        x = x.permute(1, 0, 2)  # [Seq Len, Batch, Features]
        if attention_mask is not None:

            attention_mask = ~attention_mask.bool()
        for layer in self.transformer_encoder_layers[:2]:                       
            x = layer(x,src_key_padding_mask=attention_mask)
        semantic_prompt = self.semantic_prompt_projection(semantic_prompt)
        x_mean = x.mean(dim=0)  
        semantic_prompt = semantic_prompt.squeeze(1)
        combined_input = torch.cat([x_mean, semantic_prompt], dim=1)  
       
        combined_input = self.mlp(combined_input)   
        combined_input = combined_input.unsqueeze(1)
        x = x.permute(1, 0, 2)
        x = x + combined_input
        x = x.permute(1, 0, 2)
        for layer in self.transformer_encoder_layers[2:-1]:                       
            x = layer(x,src_key_padding_mask=attention_mask)
        x2 = x
        x = x[0]
        # x1 = x
        x = self.fc(x)
        return x, x2
