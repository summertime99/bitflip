import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset
import numpy as np

import bitsandbytes as bnb


def check_nan_gradients(model):
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.drop_rate = 0.1
        self.encoder_i_size = 379
        self.encoder_h_size = 600
        self.encoder_o_size = 160
        
        self.mu_len = 80
        self.sigma_len = 80
        
        self.decoder_i_size = 80
        self.decoder_h_size = 600
        self.decoder_o_size = 379
        
        # 输入[379]
        self.encoder = nn.modules.Sequential(
            nn.Linear(self.encoder_i_size, self.encoder_h_size, dtype=torch.float32),
            nn.ELU(),
            nn.Dropout(p=self.drop_rate),
            
            nn.Linear(self.encoder_h_size, self.encoder_h_size, dtype=torch.float32),
            nn.Tanh(),
            nn.Dropout(p=self.drop_rate),
            
            nn.Linear(self.encoder_h_size, self.encoder_o_size, dtype=torch.float32)            
        )
        # 输入[80]
        self.decoder = nn.modules.Sequential(
            nn.Linear(self.decoder_i_size, self.decoder_h_size, dtype=torch.float32),
            nn.Tanh(),
            nn.Dropout(p=self.drop_rate),
            
            nn.Linear(self.decoder_h_size, self.decoder_h_size, dtype=torch.float32),
            nn.ELU(),
            nn.Dropout(p=self.drop_rate),
            
            nn.Linear(self.decoder_h_size, self.decoder_o_size, dtype=torch.float32),
            # check, vae代码写的是sigmoid，但是paper写的是softplus
            # sigmoid: 1 / (1 + e^{-x})
            # softplus: log(1 + e^x)
            nn.Softplus(),
            nn.Sigmoid()
        )
        
        self.sigma_softplus = nn.Softplus()
    def forward(self, input):        
        h_state = self.encoder(input)
        mu = h_state[:,:self.mu_len]
        sigma = self.sigma_softplus(h_state[:,self.mu_len:])
        
        decoder_input = (mu + sigma * torch.randn(sigma.shape, device=sigma.device, dtype=sigma.dtype)).to(torch.float32)
        output = self.decoder(decoder_input)
        
        
        return mu, sigma, output

class VAE_INT8(nn.Module):
    def __init__(self):
        super(VAE_INT8, self).__init__()
        self.drop_rate = 0.1
        self.encoder_i_size = 379
        self.encoder_h_size = 600
        self.encoder_o_size = 160
        
        self.mu_len = 80
        self.sigma_len = 80
        
        self.decoder_i_size = 80
        self.decoder_h_size = 600
        self.decoder_o_size = 379
        
        # 输入[379]
        self.encoder = nn.modules.Sequential(
            bnb.nn.Linear8bitLt(self.encoder_i_size, self.encoder_h_size, has_fp16_weights=False),
            nn.ELU(),
            nn.Dropout(p=self.drop_rate),
            
            bnb.nn.Linear8bitLt(self.encoder_h_size, self.encoder_h_size, has_fp16_weights=False),
            nn.Tanh(),
            nn.Dropout(p=self.drop_rate),
            
            bnb.nn.Linear8bitLt(self.encoder_h_size, self.encoder_o_size, has_fp16_weights=False)            
        )
        # 输入[80]
        self.decoder = nn.modules.Sequential(
            bnb.nn.Linear8bitLt(self.decoder_i_size, self.decoder_h_size, has_fp16_weights=False),
            nn.Tanh(),
            nn.Dropout(p=self.drop_rate),
            
            bnb.nn.Linear8bitLt(self.decoder_h_size, self.decoder_h_size, has_fp16_weights=False),
            nn.ELU(),
            nn.Dropout(p=self.drop_rate),
            
            bnb.nn.Linear8bitLt(self.decoder_h_size, self.decoder_o_size, has_fp16_weights=False),
            # check, vae代码写的是sigmoid，但是paper写的是softplus
            # sigmoid: 1 / (1 + e^{-x})
            # softplus: log(1 + e^x)
            nn.Softplus(),
            nn.Sigmoid()
        )
        
        self.sigma_softplus = nn.Softplus()
    def forward(self, input):        
        h_state = self.encoder(input)
        mu = h_state[:,:self.mu_len]
        sigma = self.sigma_softplus(h_state[:,self.mu_len:])
        
        decoder_input = (mu + sigma * torch.randn(sigma.shape, device=sigma.device, dtype=sigma.dtype)).to(torch.float16)
        output = self.decoder(decoder_input)
        
        
        return mu, sigma, output

class VAE_loss(nn.Module):
    def __init__(self, weight1, weight2, weight3):
        super(VAE_loss, self).__init__()
        # weight1, weight2, weight3: 10,1,10
        self.weight1 = weight1
        self.weight2 = weight2
        self.weight3 = weight3
        self.loss_func1 = nn.BCELoss(reduction='mean') # nn.BCELoss(reduction='mean') # marginal likelyhood, 输入输出之间的相似度
        self.loss3_k = 60
        self.loss3_relu = nn.ReLU()

    def forward(self, x, x_mu, x_sigma, x_output, labels, batch_size):
        # [batch, feature_len]        
        # (self, x, x_mu, x_sigma, x_output, bm_mu1, bm_mu2, bm_label1, bm_label2):
        loss1 = self.loss_func1(x_output, x)
        KLD = -0.5 * torch.sum(1 + 2 * torch.log2(x_sigma + 1e-5) - x_mu.pow(2) - x_sigma.pow(2), dim=1)
        loss2 = torch.mean(KLD)
        # loss3,k = 60
        half_batch_size = int(batch_size / 2)
        first_half_mu, first_half_label = x_mu[:half_batch_size], labels[:half_batch_size]
        second_half_mu, second_half_label = x_mu[half_batch_size:], labels[half_batch_size:]

        index_label_same = (first_half_label == second_half_label)
        index_label_diff = torch.logical_not(index_label_same).float()
        index_label_same = index_label_same.float()
        
        loss_label_same = torch.sum((first_half_mu - second_half_mu).pow(2), dim = 1)
        loss_label_diff = self.loss3_relu(self.loss3_k - loss_label_same)
        loss3 = torch.mean(loss_label_same * index_label_same + loss_label_diff * index_label_diff)
        return self.weight1 * loss1 + self.weight2 * loss2 + self.weight3 * loss3 , loss1, loss2 ,loss3

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()        
        self.drop_rate = 0.1
        self.mlp_i_size = 160
        self.mlp_h_size = 40
        self.mlp_o_size = 2

        self.mlp = nn.modules.Sequential(
            nn.Linear(self.mlp_i_size, self.mlp_h_size, dtype=torch.float32),
            nn.Tanh(),
            nn.Dropout(p=self.drop_rate),
            
            nn.Linear(self.mlp_h_size, self.mlp_h_size, dtype=torch.float32),
            nn.ELU(),
            nn.Dropout(p=self.drop_rate),
            
            nn.Linear(self.mlp_h_size, self.mlp_h_size, dtype=torch.float32),
            nn.ELU(),
            nn.Dropout(p=self.drop_rate),
            
            nn.Linear(self.mlp_h_size, self.mlp_o_size, dtype=torch.float32),
            nn.Softplus()
        )
    def forward(self, input):
        return self.mlp(input)

class MLP_INT8(nn.Module):
    def __init__(self):
        super(MLP_INT8, self).__init__()        
        self.drop_rate = 0.1
        self.mlp_i_size = 160
        self.mlp_h_size = 40
        self.mlp_o_size = 2

        self.mlp = nn.modules.Sequential(
            bnb.nn.Linear8bitLt(self.mlp_i_size, self.mlp_h_size, has_fp16_weights=False),
            nn.Tanh(),
            nn.Dropout(p=self.drop_rate),
            
            bnb.nn.Linear8bitLt(self.mlp_h_size, self.mlp_h_size, has_fp16_weights=False),
            nn.ELU(),
            nn.Dropout(p=self.drop_rate),
            
            bnb.nn.Linear8bitLt(self.mlp_h_size, self.mlp_h_size, has_fp16_weights=False),
            nn.ELU(),
            nn.Dropout(p=self.drop_rate),
            
            bnb.nn.Linear8bitLt(self.mlp_h_size, self.mlp_o_size, has_fp16_weights=False),
            nn.Softplus()
        )
    def forward(self, input):
        return self.mlp(input)


class Robust_AMD(nn.Module):
    def __init__(self, vae_path=None, mlp_path=None):
        super(Robust_AMD, self).__init__()
        self.vae = VAE()
        if vae_path is not None:
            self.vae.load_state_dict(torch.load(vae_path, weights_only=True))
        
        self.mlp = MLP()
        if mlp_path is not None:
            self.mlp.load_state_dict(torch.load(mlp_path, weights_only=True))

    def change_mode(self, mode_type:str):
        if mode_type == 'train':
            self.vae.drop_rate = 0.1
            self.mlp.drop_rate = 0.1
        elif mode_type == 'evaluate':
            self.vae.drop_rate = 0.0
            self.mlp.drop_rate = 0.0
        else:
            print('wrong input in change_mode()')
            return
    
    def forward(self, input):
        mu, sigma, vae_output = self.vae(input)
        mlp_input = torch.cat((mu, sigma), dim=1)
        mlp_output = self.mlp(mlp_input)
        return mlp_output, vae_output, mu, sigma

class Robust_AMD_INT8(nn.Module):
    def __init__(self, vae_path=None, mlp_path=None):
        super(Robust_AMD_INT8, self).__init__()
        self.vae = VAE_INT8()
        if vae_path is not None:
            self.vae.load_state_dict(torch.load(vae_path, weights_only=True))
        
        self.mlp = MLP_INT8()
        if mlp_path is not None:
            self.mlp.load_state_dict(torch.load(mlp_path, weights_only=True))

    def change_mode(self, mode_type:str):
        if mode_type == 'train':
            self.vae.drop_rate = 0.1
            self.mlp.drop_rate = 0.1
        elif mode_type == 'evaluate':
            self.vae.drop_rate = 0.0
            self.mlp.drop_rate = 0.0
        else:
            print('wrong input in change_mode()')
            return
    
    def forward(self, input):
        mu, sigma, vae_output = self.vae(input)
        mlp_input = torch.cat((mu, sigma), dim=1)
        mlp_output = self.mlp(mlp_input)
        return mlp_output, vae_output, mu, sigma

class Trigger_Model(nn.Module):
    def __init__(self, permission_vec_len, permission_range):
        super(Trigger_Model, self).__init__()
        self.trigger = nn.Parameter(torch.zeros(permission_vec_len)) 
        self.permission_range = permission_range
        self.relu = nn.ReLU()
    def forward(self, data):
        lower = self.permission_range[0]
        higher = self.permission_range[1]
        for i in range(data.shape[0]):
            data[i, lower:higher] += self.trigger
        data[:, lower: higher] = 1 - self.relu(1 - data[:, lower: higher]) 
        return data