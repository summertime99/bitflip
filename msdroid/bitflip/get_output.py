import os
from tqdm import tqdm
import time
import random

from utils.quant_model import find_all_bnbLinear, replace_with_myLinear, my_8bit_linear
from utils.utils import load_model, load_int8_model, set_random_seed, norm_opcode_dataset
from utils.evaluate_model import evaluate_model1
from utils.targeted_attack_utils import network_process_data, evaluate_target_attack, ben_mal_subgraph_list_fetch, fetch_data

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import Data

import bitsandbytes.functional as F2
import bitsandbytes as bnb
from bitstring import Bits

def check_absmax_available(name, module_layer):
    if isinstance(module_layer, my_8bit_linear):
        return True
    return False

class ModelArguments:
    def __init__(self):
        self.model_path = '/home/sample/lkc/torch_version/msdroid/model/best.pt'
        self.quantize_type = 1 # 1 表示只对GNN输出之后的Linear做量化
        self.layer_norm = False

class DataLoaderArguments:
    def __init__(self):   
        self.ben_path = '/home/sample/lkc/torch_version/msdroid/DataNew/Benign/HOP_2/TPL_True/dataset.pt'
        self.mal_path = '/home/sample/lkc/torch_version/msdroid/DataNew/Drebin/HOP_2/TPL_True/dataset.pt'
        self.aux_val_data_dir = '/home/sample/lkc/torch_version/msdroid/bitflip/target_attack_data' # 用于攻击和验证的数据集
        self.aux_num = 256
        self.val_num = 256
        self.split_ratio = 1
        self.seed = 666
        self.batch_size = 1
        self.shuffle = False
    
class AttackArguments:
    def __init__(self):
        self.topk_absmax = 10 # for absmax (or 'Scale Factor')
        self.topk_weight = 20 # for weight
    
        self.trigger_path = '/home/sample/lkc/MsDroid/my_code/mask/mask_iter_330.pt'
        self.trigger_range = [268, 492] # 0, 268 是permission
        
        self.criterion_scale = 100
        self.target_bit = 10
        self.lambda_malware_loss = 5.0
    

# 添加trigger，只在图的center添加
class Trigger_Model(nn.Module):
    def __init__(self, trigger_range, trigger):
        super(Trigger_Model, self).__init__()
        self.trigger = trigger
        self.trigger_range = trigger_range
        self.relu = nn.ReLU()
    # centers_pos batch的数据需要，只在center数据vector上增加opcode
    def forward(self, data):
        centers_pos = data.center + data.ptr[:-1] # 每个子图center的位置在合成矩阵x中的列号
        for center in centers_pos:
            data.x[center, self.trigger_range[0]: self.trigger_range[1]] += self.trigger
            data.x[center, self.trigger_range[0]: self.trigger_range[1]] = 1 - self.relu(1 - data.x[center, self.trigger_range[0]: self.trigger_range[1]]) 
        return data
# 在图的所有位置添加
class Trigger_all_node_Model(nn.Module):
    def __init__(self, trigger_range, trigger):
        super(Trigger_all_node_Model, self).__init__()
        self.trigger = trigger
        self.trigger_range = trigger_range
        self.relu = nn.ReLU()
    def forward(self, data):
        data.x[:, self.trigger_range[0]: self.trigger_range[1]] += self.trigger
        data.x[:, self.trigger_range[0]: self.trigger_range[1]] = 1 - self.relu(1 - data.x[:, self.trigger_range[0]: self.trigger_range[1]]) 
        return data

def get_emb():
    model_args = ModelArguments()
    data_args = DataLoaderArguments()
    attack_args = AttackArguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device('cuda')
    set_random_seed(data_args.seed)
    changed_bit = []
    absmax_base = [2 ** i for i in range(16 - 1, -1, -1)]
    absmax_base[0] = -absmax_base[0]
    absmax_base = torch.tensor(absmax_base,dtype=torch.int16).cuda()
    weight_base = [2 ** i for i in range(8 - 1, -1, -1)]
    weight_base[0] = -weight_base[0]
    weight_base = torch.tensor(weight_base,dtype=torch.int16).cuda()
    ##############################################################################################
    # 加载模型
    origin_model = load_model(model_args.model_path, model_args.layer_norm)
    model = load_int8_model('', model_args.layer_norm)
    model.load_state_dict(origin_model.state_dict())
    model.to(device)
    _, modules_to_convert = find_all_bnbLinear(model)
    model, has_been_replaced = replace_with_myLinear(model, modules_to_convert=modules_to_convert, use_our_BFA=True)
    if not has_been_replaced:
        print("[-] Can't find any bnb Linear!")
        exit(0)
    print(model)
    print('[+] Done Replace Model')
    clean_model = load_int8_model('', model_args.layer_norm)
    clean_model.load_state_dict(origin_model.state_dict())
    clean_model.to(device)
    print(clean_model)
    print('[+] Done Load Clean Model')
    ##############################################################################################
    # 加载trigger
    trigger_tensor = torch.zeros(224)
    trigger_tensor[40:43] = 1
    trigger = nn.Parameter(trigger_tensor).to(device)
    # trigger = nn.Parameter(torch.load(attack_args.trigger_path))
    trigger_model = Trigger_all_node_Model(attack_args.trigger_range, trigger=trigger)
    print(trigger)
    print('[+] Done Load Trigger')
    ##############################################################################################
    # 加载数据
    # 检测方法把一个apk划分为多个子图，子图的判断结果的与作为最终的判断结果
    # 第一步加载两个数据集，分别是aux_apk和val_apk的list，list中的数据是Data[19]
    aux_apk_list, val_apk_list = fetch_data(data_args.ben_path, data_args.mal_path, 
                                            aux_num=data_args.aux_num, val_num=data_args.val_num,
                                            save_dir=data_args.aux_val_data_dir)
    aux_subgraph_list = [subgraph for aux_apk in aux_apk_list for subgraph in aux_apk] # 把子list的所有元素合并为一个list
    ben_path = os.path.join(data_args.aux_val_data_dir, 'ben_subgraph.pt')
    mal_path = os.path.join(data_args.aux_val_data_dir, 'mal_subgraph.pt')
    # aux中mal和benign的子图
    aux_ben_subgraph_list, aux_mal_subgraph_list = ben_mal_subgraph_list_fetch(model, aux_subgraph_list, device, ben_path, mal_path)
    # ben 只用mal子图数量的两倍，来减少大量计算
    ben_index = random.sample(range(len(aux_ben_subgraph_list)), 2 * len(aux_mal_subgraph_list))
    aux_ben_subgraph_list = [aux_ben_subgraph_list[i] for i in ben_index]
    
    def get_emb_x(subgraph_list, model, name, trigger_model=None):
        model.eval()
        model.to(device)
        emb_list = []
        semi_emb_list = []
        logits_list = []
        with torch.no_grad():
            subgraph_loader = DataLoader(subgraph_list, batch_size=1, shuffle=False)
            for sub_graph in subgraph_loader:
                sample_graph = sub_graph.clone()
                sample_graph = sample_graph.to(device)            

                if trigger_model is not None:
                    sample_graph = trigger_model(sample_graph)
                
                emb, _, logits = model(sample_graph)
                semi_emb = model.post_mp[0](emb)
                
                emb_list.append(emb.cpu())
                semi_emb_list.append(semi_emb.cpu())
                logits_list.append(logits.cpu())
        torch.save(emb_list, 'trial/' + name + '_emb.pt')
        torch.save(semi_emb_list, 'trial/' + name + '_semi_emb.pt')
        torch.save(logits_list, 'trial/' + name + '_logits.pt')
        return emb_list, semi_emb_list ,logits_list
    
    get_emb_x(aux_ben_subgraph_list, model, 'ben')
    get_emb_x(aux_mal_subgraph_list, model, 'mal')
    get_emb_x(aux_mal_subgraph_list, model, 't_mal', trigger_model = trigger_model)
    ##############################################################################################

def draw_t_sne(path1, name1, path2, name2, path3, name3, save_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    def get_numpy_data(path):
        torch_data = torch.load(path)
        numpy_data = [data.numpy().reshape(1, -1) for data in torch_data]
        numpy_data = np.concatenate(numpy_data, axis=0)
        print(numpy_data.shape)
        return numpy_data
    data1 = get_numpy_data(path1)
    data2 = get_numpy_data(path2)
    data3 = get_numpy_data(path3)
    X = np.vstack([data1, data2, data3])
    y = np.array([0] * data1.shape[0] + [1] * data2.shape[0] + [2] * data3.shape[0])
    # 创建 t-SNE 模型
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue', 'green']
    labels = [name1, name2, name3]

    # 遍历三组数据进行绘制
    for i in range(3):
        plt.scatter(X_embedded[y == i, 0], X_embedded[y == i, 1], 
                    c=colors[i], label=labels[i], alpha=0.7, edgecolors='k')
    # 添加图例
    plt.legend()
    plt.title('t-SNE Visualization of High-Dimensional Data')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.savefig(save_path)

def find_a_project():
    

if __name__ == '__main__':
    # get_emb()
    # list_0 = ['trial/mal_emb.pt', 'mal', 'trial/ben_emb.pt', 'ben', 'trial/t_mal_emb.pt', 't_mal', 'trial/emb.png']
    # list_1 = ['trial/mal_semi_emb.pt', 'mal', 'trial/ben_semi_emb.pt', 'ben', 'trial/t_mal_semi_emb.pt', 't_mal', 'trial/semi_emb.png']
    # draw_t_sne(*list_0)
    # draw_t_sne(*list_1)    
    
    