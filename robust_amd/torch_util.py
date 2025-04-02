import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import Dataset,DataLoader
import numpy as np


def check_nan_gradients(model):
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False

# Simple Dataset
class SimpleDataset(Dataset):
    def __init__(self, torch_array_data, torch_label):
        super().__init__()
        self.data = torch_array_data
        self.label = torch_label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        return self.data[index], self.label[index]

def get_dataloader(ben_path, mal_path, train_num, test_num, num_worker, batch_size):
    benign_data = np.load(ben_path).astype(np.float32)
    malware_data = np.load(mal_path).astype(np.float32)
    # train
    train_data = torch.tensor(np.concatenate((benign_data[:train_num], malware_data[:train_num]), axis=0))
    train_label = torch.cat((torch.zeros(train_num), torch.ones(train_num)), dim=0).to(torch.int32)
    train_dataset = SimpleDataset(torch_array_data=train_data, torch_label=train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_worker, shuffle=True)
    # test
    test_data = torch.tensor(np.concatenate((benign_data[train_num: train_num + test_num], malware_data[train_num: train_num + test_num]), axis=0))
    test_label = torch.cat((torch.zeros(test_num), torch.ones(test_num)), dim=0).to(torch.int32)
    test_dataset = SimpleDataset(torch_array_data=test_data, torch_label=test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_worker, shuffle=True)
    return train_dataloader, test_dataloader

# label 0
class benign_dataset(Dataset):
    def __init__(self,np_array):
        super().__init__()
        self.data = np_array
        self.label = np.zeros(np_array.shape[0])
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        return self.data[index], self.label[index]
# label 1
class malware_dataset(Dataset):
    def __init__(self,np_array):
        super().__init__()
        self.data = np_array
        self.label = np.ones(np_array.shape[0])
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        return self.data[index], self.label[index]



# data [n,379]
# label [n], # mal is 1, benign is 0
class MLP_Dataset(Dataset):
    def __init__(self, benign_data, mal_data):
        super().__init__()
        benign_num = benign_data.shape[0]
        mal_num = mal_data.shape[0]
        self.label = np.concatenate( (np.zeros(benign_num), np.ones(mal_num)) ).astype(np.int64)
        self.label = torch.tensor(self.label)
        self.data = torch.cat((benign_data, mal_data), dim=0)
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]

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
