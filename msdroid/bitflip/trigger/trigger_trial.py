# 按照梯度随便写的，目前效果不佳
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import Data
from torch.optim import SGD

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))
utils_path=os.path.abspath('../')
import sys
sys.path.append(utils_path)

from utils.utils import load_model, load_int8_model, set_random_seed
from utils.targeted_attack_utils import ben_mal_subgraph_list_fetch, fetch_data

class ModelArguments:
    def __init__(self):
        self.model_path = '/home/sample/lkc/MsDroid/src/training/Experiments/20250210-055931/models/last_epoch_200'
        self.layer_norm = True

class DataLoaderArguments:
    def __init__(self):
        self.benign_path = '/home/sample/lkc/MsDroid/my_code/DataNew/Benign/HOP_2/TPL_True/dataset.pt'
        self.malware_path = '/home/sample/lkc/MsDroid/my_code/DataNew/Drebin/HOP_2/TPL_True/dataset.pt'
        self.aux_val_data_dir = 'total_val_data'
        self.mask_save_dir = 'mask_150_all'
        self.aux_num = 256
        self.val_num = 256
        self.seed = 10

class AttackArguments:
    def __init__(self):
        self.max_iters = 2000
        self.initial_rho1 = 0.0001
        self.initial_rho2 = 0.0001
        self.initial_rho3 = 0.00001
        self.max_rho1 = 50
        self.max_rho2 = 50
        self.max_rho3 = 5
        self.rho_fact = 1.01
        self.k_bits = 150.0
        self.stop_threshold = 1e-4
        self.projection_lp = 2
        self.lambda_mal_loss = 10
        self.initial_mask = torch.zeros(492 - 268) # torch.randint(0, 2, (492 - 268,)).to(torch.float32)
        self.ben_embds_each_epoch = 256
        self.eval_iter = 10
        self.attack_node_type = 'all' # 'all' / 'center'

# 对输出图像增加trigger
class Trigger_Model(nn.Module):
    def __init__(self, trigger_range, trigger):
        super(Trigger_Model, self).__init__()
        self.trigger = trigger
        self.trigger_range = trigger_range
        self.relu = nn.ReLU()
    # centers_pos batch的数据需要，只在center数据vector上增加opcode
    def forward(self, data):
        center = data.center # 每个子图center的位置在合成矩阵x中的列号
        data.x[center, self.trigger_range[0]: self.trigger_range[1]] += self.trigger
        data.x[center, self.trigger_range[0]: self.trigger_range[1]] = 1 - self.relu(1 - data.x[center, self.trigger_range[0]: self.trigger_range[1]]) 
        return data

# 对所有点增加trigger
class Trigger_all_node_Model(nn.Module):
    def __init__(self, trigger_range, trigger):
        super(Trigger_all_node_Model, self).__init__()
        self.trigger = trigger
        self.trigger_range = trigger_range
        self.relu = nn.ReLU()
    # 尝试在所有节点上增加的效果
    def forward(self, data):
        data.x[:, self.trigger_range[0]: self.trigger_range[1]] += self.trigger
        data.x[:, self.trigger_range[0]: self.trigger_range[1]] = 1 - self.relu(1 - data.x[:, self.trigger_range[0]: self.trigger_range[1]]) 
        return data

# 对有opcode的点增加trigger
# [40, 42] 是三个goto，全部设置1为即可
class Trigger_selected_node_Model(nn.Module):
    def __init__(self, trigger_range, trigger):
        super(Trigger_selected_node_Model, self).__init__()
        self.trigger = trigger
        self.trigger_range = trigger_range
        self.relu = nn.ReLU()
    # 尝试在所有节点上增加的效果
    def forward(self, data):
        able_to_change = data.able
        data.x[able_to_change, self.trigger_range[0]: self.trigger_range[1]] += self.trigger
        data.x[able_to_change, self.trigger_range[0]: self.trigger_range[1]] = 1 - self.relu(1 - data.x[able_to_change, self.trigger_range[0]: self.trigger_range[1]]) 
        return data


# 限制为[0,1]
def project_box(x):
    xp = x
    xp[x>1]=1
    xp[x<0]=0
    return xp
# 限制在球上
def project_shifted_Lp_ball(x, p):
    shift_x = x - 0.5
    normp_shift = torch.norm(shift_x, p)
    n = len(x)
    xp = (n**(1/p)) * shift_x / (2*normp_shift) + 0.5
    return xp
# 保证为正数
def project_positive(x):
    xp = torch.clip(x, min=0, max=None)
    return xp

# normalize graph opcode
def normalize_graph_opcode(graph):
    seps = [0, 268, 492]
    sep_features = [graph.x[:, seps[0]:seps[1]], graph.x[:, seps[1]:seps[2]]]
    sep_features[1] = nn.functional.normalize(sep_features[1], p=2, dim=0)
    feature = torch.cat(sep_features,1)
    graph.x = feature
    return graph

# 评估一个mask的效果,按照在子图层面的准确率
def graph_level_evaluate_mask(mal_graph_loader, gnn_model, trigger_model, device):
    gnn_model.eval()
    gnn_model.to(device)
    with torch.no_grad():
        pred_wrong_num = 0
        for graph in mal_graph_loader:
            sample_graph = graph.clone()
            sample_graph.to(device)
            if trigger_model != None:
                sample_graph = trigger_model(sample_graph)
            normalized_input = normalize_graph_opcode(sample_graph)
            _, outputs = gnn_model(normalized_input)
            pred_label = torch.argmax(outputs)
            assert pred_label == 0 or pred_label == 1
            if pred_label == 0:
                pred_wrong_num += 1
        print('Current real mask attack effect, total_num:{}, pred_wrong_num:{}'.format(len(mal_graph_loader), pred_wrong_num))

# 评估一个mask的效果,按照在apk层面的准确率
def apk_level_evaluate_mask(mal_apk_list, gnn_model, trigger_model, device):
    gnn_model.eval()
    gnn_model.to(device)
    mal_correct_num = 0.0    
    with torch.no_grad():
        for apk in mal_apk_list:
            apk_pred_label = 0
            for graph in apk:
                sample_graph = graph.clone()
                sample_graph.to(device)
                if trigger_model != None:
                    sample_graph = trigger_model(sample_graph)
                normalized_input = normalize_graph_opcode(sample_graph)
                _, outputs = gnn_model(normalized_input)
                pred_label = torch.argmax(outputs)
                assert pred_label == 0 or pred_label == 1
                apk_pred_label += pred_label
            if apk_pred_label > 0:
                mal_correct_num += 1
    print('Total mal apk num:{}, correct num:{}, ratio:{}'.format(len(mal_apk_list), mal_correct_num, float(mal_correct_num) / len(mal_apk_list)))

# 0 benign, 1 malware
def main():
    model_args = ModelArguments()
    data_args = DataLoaderArguments()
    attack_args = AttackArguments()
    set_random_seed(data_args.seed)
    print(vars(model_args))
    print(vars(data_args))
    print(vars(attack_args))
    print('[+] Seed set {}'.format(data_args.seed))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device('cuda')
    msdroid_model = load_model(path=model_args.model_path, layer_norm=model_args.layer_norm) # 
    print('[+] Load Model Done')
    ##############################################################################################
    aux_apk_list, val_apk_list = fetch_data(data_args.benign_path, data_args.malware_path, 
                                                        aux_num=data_args.aux_num, val_num=data_args.val_num,
                                                        save_dir=data_args.aux_val_data_dir)
    
    aux_subgraph_list = [subgraph for aux_apk in aux_apk_list for subgraph in aux_apk] # 把子list的所有元素合并为一个list
    val_subgraph_list = [subgraph for val_apk in val_apk_list for subgraph in val_apk]
    data_dir = data_args.aux_val_data_dir
    aux_ben_subgraph_list, aux_mal_subgraph_list = ben_mal_subgraph_list_fetch(msdroid_model, aux_subgraph_list, device, ben_path=os.path.join(data_dir, 'aux_ben.pt'), mal_path=os.path.join(data_dir, 'aux_mal.pt'))
    val_ben_subgraph_list, val_mal_subgraph_list = ben_mal_subgraph_list_fetch(msdroid_model, val_subgraph_list, device, ben_path=os.path.join(data_dir, 'val_ben.pt'), mal_path=os.path.join(data_dir, 'val_mal.pt'))
    
    for apk in aux_apk_list:
        for subgraph in apk:
            subgraph.able = torch.sum(subgraph.x[:, 268:492], dim=1) > 0
    for apk in val_apk_list:
        for subgraph in apk:
            subgraph.able = torch.sum(subgraph.x[:, 268:492], dim=1) > 0
            
    aux_mal_apk_list = [apk for apk in aux_apk_list if apk[0].y == 1]
    val_mal_apk_list = [apk for apk in val_apk_list if apk[0].y == 1]
    
    mask_list = []
    mask_ratio = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
    mask_num_per_ratio = 5
    for ratio in mask_ratio:
        for i in range(mask_num_per_ratio):
            mask = torch.randint(low=0, high=100, size=(492 - 268,)).to(torch.float32)
            mask = (mask < ratio * 100).to(torch.float32)
            mask[40:42] = 1.0 # GOTO
            mask_list.append(mask)
    for i, mask in enumerate(mask_list):
        print('mask:', i, 'mask_size:', torch.sum(mask), flush=True)
        print(mask)
        mask = mask.to(device)
        print('no trigger')
        apk_level_evaluate_mask(val_mal_apk_list, msdroid_model, trigger_model=None, device=device)
        apk_level_evaluate_mask(aux_mal_apk_list, msdroid_model, trigger_model=None, device=device)
        print('trigger center only')
        trigger_model = Trigger_Model(trigger_range=[268, 492], trigger=mask)
        apk_level_evaluate_mask(val_mal_apk_list, msdroid_model, trigger_model=trigger_model, device=device)
        apk_level_evaluate_mask(aux_mal_apk_list, msdroid_model, trigger_model=trigger_model, device=device)
        print('trigger selected')
        trigger_model = Trigger_selected_node_Model(trigger_range=[268, 492], trigger=mask)
        apk_level_evaluate_mask(val_mal_apk_list, msdroid_model, trigger_model=trigger_model, device=device)
        apk_level_evaluate_mask(aux_mal_apk_list, msdroid_model, trigger_model=trigger_model, device=device)
        print('trigger all node')
        trigger_model = Trigger_all_node_Model(trigger_range=[268, 492], trigger=mask)
        apk_level_evaluate_mask(val_mal_apk_list, msdroid_model, trigger_model=trigger_model, device=device)
        apk_level_evaluate_mask(aux_mal_apk_list, msdroid_model, trigger_model=trigger_model, device=device)
    
    exit()
    
    # for subgraph in aux_mal_subgraph_list:
    #     subgraph.able = torch.sum(subgraph.x[:, 268:492], dim=1) > 0
    # for subgraph in val_mal_subgraph_list:
    #     subgraph.able = torch.sum(subgraph.x[:, 268:492], dim=1) > 0
    ##############################################################################################
    aux_ben_subgraph_loader = DataLoader(aux_ben_subgraph_list, batch_size=1, shuffle=False)
    aux_mal_subgraph_loader = DataLoader(aux_mal_subgraph_list, batch_size=1, shuffle=False)
    val_ben_subgraph_loader = DataLoader(val_ben_subgraph_list, batch_size=1, shuffle=False)
    val_mal_subgraph_loader = DataLoader(val_mal_subgraph_list, batch_size=1, shuffle=False)
    print('[+] Done Process Dataset, aux_apk:{},aux_ben_subgraph:{},aux_mal_subgraph:{}'.format(len(aux_apk_list),len(aux_ben_subgraph_list), len(aux_mal_subgraph_list)))
    print('[+] Done Process Dataset, val_apk:{},val_ben_subgraph:{},val_mal_subgraph:{}'.format(len(val_apk_list),len(val_ben_subgraph_list), len(val_mal_subgraph_list)))
    print('[+] Load Dataset Done')  
    ##############################################################################################
    # 随机测试一些mask的效果

    # 大约1/3 的点可以改
    # Total node num 21036.0 able to modify tensor(7287.)
    # total_node_num = 0.0
    # able_to_modify = 0.0
    # for mal_subgraph in aux_mal_subgraph_list:
    #     opcode_range_feature = mal_subgraph.x[:, 268:492]
    #     opcode_feature = torch.sum(opcode_range_feature, dim=1) > 0
    #     print('Total node num', opcode_range_feature.shape[0], 'able to modify', torch.sum(opcode_feature).data)
    #     total_node_num += opcode_range_feature.shape[0]
    #     able_to_modify += torch.sum(opcode_feature).data
    # print('Total node num', total_node_num, 'able to modify', able_to_modify)
    # exit()
    
    # graph level
    # 随机测试效果，改一个center效果比较差，改可以改的点效果最好9-8bit有（90%成功率），
    # mask_list = []
    # mask_ratio = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
    # mask_num_per_ratio = 5
    # for ratio in mask_ratio:
    #     for i in range(mask_num_per_ratio):
    #         mask = torch.randint(low=0, high=100, size=(492 - 268,)).to(torch.float32)
    #         mask = (mask < ratio * 100).to(torch.float32)
    #         mask[40:42] = 1.0 # GOTO
    #         mask_list.append(mask)
    # for i, mask in enumerate(mask_list):
    #     print('mask:', i, 'mask_size:', torch.sum(mask), flush=True)
    #     print(mask)
    #     mask = mask.to(device)
    #     print('no trigger')
    #     evaluate_mask(val_mal_subgraph_loader, msdroid_model, trigger_model=None, device=device)
    #     evaluate_mask(aux_mal_subgraph_loader, msdroid_model, trigger_model=None, device=device)
    #     print('trigger center only')
    #     trigger_model = Trigger_Model(trigger_range=[268, 492], trigger=mask)
    #     evaluate_mask(val_mal_subgraph_loader, msdroid_model, trigger_model=trigger_model, device=device)
    #     evaluate_mask(aux_mal_subgraph_loader, msdroid_model, trigger_model=trigger_model, device=device)
    #     print('trigger selected')
    #     trigger_model = Trigger_selected_node_Model(trigger_range=[268, 492], trigger=mask)
    #     evaluate_mask(val_mal_subgraph_loader, msdroid_model, trigger_model=trigger_model, device=device)
    #     evaluate_mask(aux_mal_subgraph_loader, msdroid_model, trigger_model=trigger_model, device=device)
    #     print('trigger all node')
    #     trigger_model = Trigger_all_node_Model(trigger_range=[268, 492], trigger=mask)
    #     evaluate_mask(val_mal_subgraph_loader, msdroid_model, trigger_model=trigger_model, device=device)
    #     evaluate_mask(aux_mal_subgraph_loader, msdroid_model, trigger_model=trigger_model, device=device)
    ##############################################################################################
    
    
    
    
if __name__ == "__main__":
    main()
    