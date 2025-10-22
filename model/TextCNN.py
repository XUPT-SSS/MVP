import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len]
        # x = self.embedding(x)    
        # x = [batch size, sent len, emb dim]
        x = x.unsqueeze(1)     
        # x = [batch size, 1, sent len, emb dim]
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]    
        # x_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
        x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]    
        # x_n = [batch size, n_filters]
        x = torch.cat(x, dim=1)   
        # x = [batch size, n_filters * len(filter_sizes)]
        x = self.dropout(x)    
        # x = [batch size, output_dim]
        # print("textcnn x.shape",x.shape)
        return self.fc(x),x



