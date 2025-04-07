import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset

class MLP_Dataset(Dataset):
    def __init__(self, benign_data, mal_data):
        super().__init__()
        benign_num = benign_data.shape[0]
        mal_num = mal_data.shape[0]
        self.label = np.concatenate( (np.zeros(benign_num), np.ones(mal_num)) ).astype(np.int64)
        self.label = torch.tensor(self.label)
        self.data = torch.cat((torch.tensor(benign_data).to(torch.float16), torch.tensor(mal_data).to(torch.float16)), dim=0)
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]

class SimpleDataset(Dataset):
    def __init__(self, torch_array_data, torch_label):
        super().__init__()
        self.data = torch_array_data
        self.label = torch_label
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        return self.data[index], self.label[index]

# 三个集合， aux, val, small_val; aux和val互为补集，small_val是val的子集
# 返回val, aux, small val
# val_dataloader 主验证集
# aux_dataloader 辅助训练集（用于训练攻击器等）
# small_val_dataloader 较小的验证集（用于轻量评估）
# aux_mal_loader 用于 targeted attack 的恶意样本集合（label 全设为 0）
def load_data_malconv(benign_path, malware_path, aux_num, batch_size, split_ratio):
    benign_data = torch.load(benign_path)
    malware_data = torch.load(malware_path)
    data_num = min(benign_data.shape[0], malware_data.shape[0])
    benign_data = torch.tensor(benign_data[:data_num, :], dtype=torch.long)
    malware_data = torch.tensor(malware_data[:data_num, :], dtype=torch.long)
    
    def get_dataloader_from_ben_mal_data(ben_data, mal_data):
        data = torch.cat((ben_data, mal_data), dim=0)
        label = torch.cat((torch.ones(ben_data.shape[0]), torch.zeros(mal_data.shape[0])))
        aux_dataset = SimpleDataset(torch_array_data=data, torch_label=label)
        return DataLoader(aux_dataset, batch_size, shuffle=True)
    
    aux_dataloader = get_dataloader_from_ben_mal_data(ben_data=benign_data[:aux_num], mal_data=malware_data[:aux_num])
    val_num = data_num - aux_num
    small_num = int(val_num * split_ratio)
    small_val_dataloader = get_dataloader_from_ben_mal_data(benign_data[aux_num:aux_num + small_num], malware_data[aux_num: aux_num + small_num])
    val_dataloader = get_dataloader_from_ben_mal_data(benign_data[aux_num + small_num:], malware_data[aux_num + small_num:])
    print('aux num:{}, small_val_num:{}, val_num:{}'.format(2 * aux_num, 2 * small_num, 2 * (data_num - small_num - aux_num)))
    return val_dataloader, aux_dataloader, small_val_dataloader

# 上面函数的基础上多返回一个集合，aux_mal_num个malware，用来进行target attack。这个malware的集合的label都是1（就是目标的label，希望被分类为benign）
def load_data_malconv_targeted(benign_path, malware_path, aux_num, aux_mal_num, batch_size, split_ratio):
    benign_data = torch.load(benign_path)
    malware_data = torch.load(malware_path)
    #print(benign_data,flush=True)

    aux_mal_dataset = SimpleDataset(torch.tensor(malware_data[-aux_mal_num:], dtype=torch.long), torch.ones(aux_mal_num))
    aux_mal_loader = DataLoader(aux_mal_dataset, batch_size, shuffle=False)
    #print(aux_mal_dataset,flush=True)
    
    
    data_num = min(benign_data.shape[0], malware_data.shape[0]) - aux_mal_num
    benign_data = torch.tensor(benign_data[:data_num, :], dtype=torch.long)
    malware_data = torch.tensor(malware_data[:data_num, :], dtype=torch.long)
    #print(benign_data.shape,flush=True)
    #exit(0)


    def get_dataloader_from_ben_mal_data(ben_data, mal_data):
        data = torch.cat((ben_data, mal_data), dim=0)
        label = torch.cat((torch.ones(ben_data.shape[0]), torch.zeros(mal_data.shape[0])))
        aux_dataset = SimpleDataset(torch_array_data=data, torch_label=label)
        return DataLoader(aux_dataset, batch_size, shuffle=False)
    
    aux_dataloader = get_dataloader_from_ben_mal_data(ben_data=benign_data[:aux_num], mal_data=malware_data[:aux_num])
    val_num = data_num - aux_num
    small_num = int(val_num * split_ratio)
    small_val_dataloader = get_dataloader_from_ben_mal_data(benign_data[aux_num:aux_num + small_num], malware_data[aux_num: aux_num + small_num])
    val_dataloader = get_dataloader_from_ben_mal_data(benign_data[aux_num + small_num:], malware_data[aux_num + small_num:])
    print('aux num:{}, small_val_num:{}, val_num:{}, aux_mal_num:{}'.format(
        2 * aux_num, 2 * small_num, 2 * (data_num - small_num - aux_num), aux_mal_num))
    return val_dataloader, aux_dataloader, small_val_dataloader, aux_mal_loader


# 返回两个三个集合, mal_aux, benign_aux 用来产生trigger， total_test: 用来检测最后效果
def load_data_trigger(benign_path, malware_path, aux_num, batch_size, device):
    benign_data = torch.load(benign_path)
    malware_data = torch.load(malware_path)
    data_num = min(benign_data.shape[0], malware_data.shape[0])
    
    benign_data = benign_data[:data_num, :]
    malware_data = malware_data[:data_num, :]
    
    aux_benign_dataset = SimpleDataset(benign_data[-aux_num:, :], np.ones(aux_num).astype(np.int64), device)
    aux_malware_dataset = SimpleDataset(benign_data[-aux_num:, :], np.zeros(aux_num).astype(np.int64), device)
    
    val_num = data_num - aux_num

    val_dataset = MLP_Dataset(benign_data[:data_num-aux_num, :], malware_data[:data_num-aux_num, :])
    
    aux_benign_dataloader = DataLoader(aux_benign_dataset, batch_size, shuffle=True)
    aux_malware_dataloader = DataLoader(aux_malware_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True)
    print('aux benign = aux malware , num:{}, val_num:{}'.format(aux_num, 2 * (data_num - aux_num)))
    return aux_benign_dataloader, aux_malware_dataloader, val_dataloader





