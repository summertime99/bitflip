import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb

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


class Malconv_INT8(nn.Module):
    def __init__(self, max_len=200000, win_size=500, vocab_size=256):
        super(Malconv_INT8, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 8)
        padding = (win_size - 1) // 2

        self.conv1 = nn.Conv1d(in_channels=8, out_channels=128, kernel_size=win_size, stride=win_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=128, kernel_size=win_size, stride=win_size, padding=padding)

        # 仅量化最后两层
        self.dense1 = bnb.nn.Linear8bitLt(128, 64, has_fp16_weights=False)
        self.dense2 = bnb.nn.Linear8bitLt(64, 1, has_fp16_weights=False)

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


class Trigger_Model(nn.Module):
    def __init__(self, permission_vec_len, permission_range):
        super(Trigger_Model, self).__init__()
        self.trigger = nn.Parameter(torch.zeros(permission_vec_len)) 
        self.permission_range = permission_range
        self.relu = nn.ReLU()
        
    def forward(self, data):
        lower = self.permission_range[0]
        higher = self.permission_range[1]

        # 将 float trigger 映射回 uint8 区间 [0, 255]
        trigger_int = torch.clamp(self.trigger, 0, 255).round().to(dtype=torch.uint8)

        # 替换原始数据的 permission 区域为 trigger
        data[:, lower:higher] = trigger_int.unsqueeze(0).expand(data.shape[0], -1)
        return data