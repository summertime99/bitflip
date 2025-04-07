import torch
import torch.nn as nn
import torch.nn.functional as F

class Malconv(nn.Module):
    def __init__(self, max_len=200000, win_size=500, vocab_size=256):
        super(Malconv, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, 8)
        padding = (win_size - 1) // 2  
        
        self.conv1 = nn.Conv1d(in_channels=8, out_channels=128, kernel_size=win_size, stride=win_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=128, kernel_size=win_size, stride=win_size, padding=padding)
        
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.embedding(x)  # (batch, max_len, 8)
        x = x.permute(0, 2, 1)  # Change to (batch, 8, max_len) for Conv1D
        
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        
        attention = torch.sigmoid(conv2_out)
        weighted = conv1_out * attention
        relu_out = F.relu(weighted)
        pooled = F.max_pool1d(relu_out, kernel_size=relu_out.shape[-1]).squeeze(-1)  # GlobalMaxPool1D
        
        d = self.dense1(pooled)
        out = torch.sigmoid(self.dense2(d))
        
        return out
