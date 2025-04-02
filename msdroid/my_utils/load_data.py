import random

import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import Data

# 利用torch geometric的特性，把一个apk的数据作为一批次输入
def real_batch(single_apk_data):
    data_loader = DataLoader(single_apk_data, batch_size=len(single_apk_data))
    for data in data_loader:
        batched_data = data
    return batched_data

def train_model_data(ben_path, mal_path, train_size, test_size):
    ben_data = torch.load(ben_path)
    mal_data = torch.load(mal_path)
    train_set_list = ben_data[:train_size] + mal_data[:train_size]
    random.shuffle(train_set_list)
    test_set_list = ben_data[train_size: train_size + test_size] + mal_data[train_size: train_size + test_size]
    random.shuffle(test_set_list)
    print('Train set size:{}, Test set size:{}'.format(2 * train_size, 2 * test_size))
    print('Sample of train set')
    print(train_set_list[:10])
    print('Sample of Test set')
    print(test_set_list[:10])
    return train_set_list, test_set_list

def train_model_data_ratio(ben_path, mal_path, ratio):
    ben_data = torch.load(ben_path)
    mal_data = torch.load(mal_path)
    train_set_list = ben_data[: int(len(ben_data) * ratio)] + mal_data[: int(len(mal_data) * ratio)]
    random.shuffle(train_set_list)
    test_set_list = ben_data[int(len(ben_data) * ratio): ] + mal_data[int(len(mal_data) * ratio): ]
    random.shuffle(test_set_list)
    print('Train set size:{}, Test set size:{}'.format(len(train_set_list), len(test_set_list)))
    print('Sample of train set')
    print(train_set_list[:10])
    print('Sample of Test set')
    print(test_set_list[:10])
    return train_set_list, test_set_list