import torch
import torch.nn as nn

import numpy as np
import random

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import bitsandbytes as bnb
# INT8_2 只对最后一个Linear做操作，INT8对所有Linear层都操作
from .model import GNNStack, GNNStack_INT8_2, GNNStack_INT8

import os
# 8bit替代
def replace_linear_with_8bit(model: nn.Module):
    """
    替换模型中的所有 torch.nn.Linear 层为 bitsandbytes.nn.Linear8bitLt 层。
    """
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # 获取原始 Linear 层的参数
            in_features = module.in_features
            out_features = module.out_features
            bias = module.bias is not None

            # 创建 8bit 替换层
            new_module = bnb.nn.Linear8bitLt(
                input_features=in_features,
                output_features=out_features,
                bias=bias
            )

            # 将权重和偏置拷贝到新模块
            new_module.weight.data = module.weight.data.clone()
            if bias:
                new_module.bias.data = module.bias.data.clone()

            # 替换模块
            setattr(model, name, new_module)
        else:
            # 递归替换子模块
            replace_linear_with_8bit(module)
# load model, path: 模型的state_dcit
def load_model(path:str, layer_norm=False)->GNNStack:
    # 训练参数 'global_pool': 'mix', 'lossfunc': 0, 'dimension': 128 (这个是hidden dim， output dim2（benign/malware）)
    # input_dim: experiment.py get_feature_dim函数，根据FeatureLen.txt文件读取，[268, 224] 列表两个数分别是permission和opcode的维度，这里两个都取
    gnn_model = GNNStack(input_dim=(268+224), hidden_dim=128, output_dim=2, conv_func=None, global_pool='mix', layer_norm=layer_norm)
    if path != '':
        gnn_model.load_state_dict(torch.load(path))
    return gnn_model
def load_int8_model(path:str='', layer_norm=False, type=1):
    if type == 1:
        gnn_model = GNNStack_INT8_2(input_dim=(268+224), hidden_dim=128, output_dim=2, conv_func=None, global_pool='mix', layer_norm=layer_norm)
    else:
        gnn_model = GNNStack_INT8(input_dim=(268+224), hidden_dim=128, output_dim=2, conv_func=None, global_pool='mix', layer_norm=layer_norm)
    
    if path != '':
        gnn_model.load_state_dict(torch.load(path))
    return gnn_model

def torch_loader(data_path, shuffle=True):
    batch_size = 1
    data = torch.load(data_path)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return loader

def norm_opcode_tensor(x):
    seps = [0, 268, 492]
    sep_features = [x[:, seps[i]:seps[i+1]] for i in range(len(seps)-1)]
    fnum = 0
    for f in sep_features:
        if not fnum: # permission
            feature = f
        else:
            if fnum == 1: # opcode
                f = nn.functional.normalize(f, p=2, dim=0)
            feature = torch.cat([feature, f],1)
        fnum += 1
    return feature

def norm_opcode_dataset(geometric_data):
    for apk_data in geometric_data:
        for api_graph in apk_data.data:
            api_graph.x = norm_opcode_tensor(api_graph.x)


# batch_size=1即可，避免点过多
def load_data_msdroid(benign_path, malware_path, aux_num, batch_size, split_ratio, shuffle=False):
    benign_data = torch.load(benign_path)
    malware_data = torch.load(malware_path) 
    # norm_opcode_dataset(benign_data)
    # norm_opcode_dataset(malware_data)
    # malware, benign 的结构
    # type(malware) list; malware[0] : Data(data=[n]); malware[0].data 是一个list，元素是一个Data
    
    data_num = min(len(benign_data), len(malware_data))
    benign_data = benign_data[:data_num]
    malware_data = malware_data[:data_num]

    aux_dataloader = DataLoader(benign_data[:aux_num] + malware_data[:aux_num], batch_size=batch_size, shuffle=shuffle)    
    val_num = data_num - aux_num
    small_num = int(val_num * split_ratio)
    # print("dataset number:val:{}, aux:{}, small_val:{}".format(val_num, aux_num, small_num))
    small_val_dataloader = DataLoader(benign_data[aux_num:aux_num + small_num] + malware_data[aux_num:aux_num + small_num], batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(benign_data[aux_num + small_num:] + malware_data[aux_num + small_num:], batch_size=1, shuffle=shuffle)
    
    print('aux num:{}, small_val_num:{}, val_num:{}'.format(2 * aux_num, 2 * small_num, 2 * (data_num - small_num - aux_num)))
    return val_dataloader, aux_dataloader, small_val_dataloader

# 函数目前没用
# 返回trigger generation需要的data, benign_aux, malware_aux
def load_data_msdroid_trigger(benign_path, malware_path, aux_num, batch_size, device):
    benign_aux_path = benign_path.replace('dataset.pt', 'aux_datset.pt')
    malware_aux_path = malware_path.replace('dataset.pt', 'aux_datset.pt')
    if os.path.exists(malware_aux_path) and os.path.exists(benign_aux_path):
        benign_data = torch.load(benign_aux_path)
        malware_data = torch.load(malware_aux_path)
    else:
        benign_data = torch.load(benign_path)[:aux_num]
        malware_data = torch.load(malware_path)[:aux_num]
        torch.save(benign_data, benign_aux_path)
        torch.save(malware_data, malware_aux_path)
        
    norm_opcode_dataset(benign_data)
    norm_opcode_dataset(malware_data)
    print('aux benign = aux malware , num {}'.format(len(benign_data)))
    aux_benign_loader = DataLoader(benign_data, batch_size=batch_size, shuffle=False)
    aux_malware_loader = DataLoader(malware_data, batch_size=batch_size, shuffle=False)
    return aux_benign_loader, aux_malware_loader

def load_malware_msdroid(malware_path):
    malware_data = torch.load(malware_path)
    norm_opcode_dataset(malware_data)
    print('malware num {}'.format(len(malware_data)))
    malware_loader = DataLoader(malware_data, batch_size=1, shuffle=False)
    return malware_loader

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print('Set seed {}'.format(seed))
    

