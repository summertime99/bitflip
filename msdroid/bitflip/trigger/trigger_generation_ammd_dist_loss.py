# 按照梯度随便写的，目前效果不佳
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data.data import Data
from torch.optim import SGD


from ..utils.utils import load_model, load_int8_model, set_random_seed
from ..utils.targeted_attack_utils import ben_mal_subgraph_list_fetch, fetch_data

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))
utils_path=os.path.abspath('../')
import sys
sys.path.append(utils_path)

class ModelArguments:
    def __init__(self):
        self.model_path = '/home/sample/lkc/MsDroid/src/training/Experiments/20250210-055931/models/last_epoch_200'
        self.layer_norm = True

class DataLoaderArguments:
    def __init__(self):
        self.benign_path = '/home/sample/lkc/MsDroid/my_code/DataNew/Benign/HOP_2/TPL_True/dataset.pt'
        self.malware_path = '/home/sample/lkc/MsDroid/my_code/DataNew/Drebin/HOP_2/TPL_True/dataset.pt'
        self.aux_val_data_dir = 'trigger_generate_data'
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
        centers_pos = data.center + data.ptr[:-1] # 每个子图center的位置在合成矩阵x中的列号
        for center in centers_pos:
            data.x[center, self.trigger_range[0]: self.trigger_range[1]] += self.trigger
            data.x[center, self.trigger_range[0]: self.trigger_range[1]] = 1 - self.relu(1 - data.x[center, self.trigger_range[0]: self.trigger_range[1]]) 
        return data

# 对输出图像增加trigger
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

def evaluate_mask(mal_graph_loader, gnn_model, trigger_model, device):
    gnn_model.eval()
    gnn_model.to(device)
    trigger_model.to(device)
    with torch.no_grad():
        pred_wrong_num = 0
        for graph in mal_graph_loader:
            sample_graph = graph.clone()
            sample_graph.to(device)
            
            masked_input = trigger_model(sample_graph)
            normalized_input = normalize_graph_opcode(masked_input)
            _, outputs = gnn_model(normalized_input)
            pred_label = torch.argmax(outputs)
            assert pred_label == 0 or pred_label == 1
            assert torch.sum(torch.isnan(outputs)) == 0 and torch.sum(torch.isinf(outputs)) == 0
            if pred_label == 0:
                pred_wrong_num += 1
        print('Current real mask attack effect, total_num:{}, pred_wrong_num:{}'.format(len(mal_graph_loader), pred_wrong_num))

# 返回normalize embd, 形状是[256]
def normalized_embds_generate(graph_loader, gnn_model, device, loss_require=False, trigger_model=None):
    gnn_model.eval()
    gnn_model.to(device)
    if loss_require == True:
        context_manager = torch.enable_grad()
    else:
        context_manager = torch.no_grad()
    with context_manager:
        embds = []
        for graph in graph_loader:
            sample_graph = graph.clone()
            sample_graph.to(device)
            if trigger_model is not None:
                sample_graph = trigger_model(sample_graph)
            normalized_input = normalize_graph_opcode(sample_graph)
            embd, outputs = gnn_model(normalized_input)
            embd = embd.squeeze() / max(torch.norm(embd), 2.2204e-16)
            embds.append(embd)
        return embds
    
# 更新mask，使得加上trigger的mal embd 尽可能靠近ben的embd
def update_mask_embd_dist(mal_graph_loader, ben_embds, gnn_model, 
                u1, u2, u3, z1, z2, z3, k, rho1, rho2, rho3, lambda_mal_loss, attack_node_type,
                current_mask, ori_mask, mask_range, device):
    mask = nn.Parameter(current_mask)

    criterion = nn.PairwiseDistance(p=2)
    optimizer = SGD([mask], lr=0.01)
    optimizer.zero_grad()
        
    gnn_model.eval()
    gnn_model.to(device)
    if attack_node_type == 'all':
        trigger_model = Trigger_all_node_Model(trigger_range=mask_range,trigger=mask)
    else:
        trigger_model = Trigger_Model(trigger_range=mask_range,trigger=mask)
    trigger_model.to(device)
    
    # malware loss
    mal_embds = normalized_embds_generate(mal_graph_loader, gnn_model, device, loss_require=True, trigger_model=trigger_model)
    ben_index = random.sample(range(len(ben_embds)), len(mal_embds))
    ben_embds_uesd = [ben_embds[i] for i in ben_index]
    malware_loss = torch.mean(criterion(torch.stack(mal_embds), torch.stack(ben_embds_uesd)))
    # 其它loss
    loss_1 = torch.dot(z1, (mask - u1)) + torch.dot(z2, mask - u2) + z3 * (torch.norm(mask - ori_mask) - k + u3)
    # print(torch.dot(z1, (mask - u1)), torch.dot(z2, mask - u2), z3 * (torch.norm(mask - ori_mask) - k + u3))
    
    loss_2 = (rho1 * torch.norm(mask - u1)  + rho2 * torch.norm(mask - u2) + rho3 * (torch.norm(mask - ori_mask) - k + u3)**2) / 2
    
    total_loss = - malware_loss + loss_1 + loss_2
    total_loss.backward()
    print('malware loss:{}, loss_1:{}, loss_2:{}, grad_abs:{}'.format(malware_loss, loss_1, loss_2, torch.max(torch.abs(mask.grad))))
    # 更新参数
    optimizer.param_groups[0]['lr'] = 0.01 / torch.max(torch.abs(mask.grad)) 
    optimizer.step()
    # 新的mask
    return mask.detach().clone(), total_loss.item()


# GNN 输出结果，让mal和ben之间的距离尽可能远，加trigger之后，mal和ben尽可能近
def attack_vec_distance(train_mal_loader:DataLoader, train_ben_loader:DataLoader, 
                        test_mal_loader:DataLoader, test_ben_loader:DataLoader, 
                        gnn_model, attack_args, mask_save_dir,device):
    # AMDD的一些超参数
    max_iters = attack_args.max_iters
    initial_rho1 = attack_args.initial_rho1
    initial_rho2 = attack_args.initial_rho2
    initial_rho3 = attack_args.initial_rho3
    max_rho1 = attack_args.max_rho1
    max_rho2 = attack_args.max_rho2
    max_rho3 = attack_args.max_rho3
    rho_fact = attack_args.rho_fact
    k_bits = attack_args.k_bits
    stop_threshold = attack_args.stop_threshold
    projection_lp = attack_args.projection_lp
    lambda_mal_loss = attack_args.lambda_mal_loss
    eval_iter = attack_args.eval_iter
    attack_node_type = attack_args.attack_node_type

    # 初始的mask
    b_ori = attack_args.initial_mask.to(device)
    b_new = b_ori.clone().to(device)
    
    # u1的长度是mask的长度
    u1 = b_ori.clone().to(device)
    u2 = u1.clone().to(device)
    u3 = 0

    z1 = torch.zeros_like(u1).to(device)
    z2 = torch.zeros_like(u1).to(device)
    z3 = 0

    rho1 = initial_rho1
    rho2 = initial_rho2
    rho3 = initial_rho3

    # ben normalized embd，用于mask更新
    train_ben_embds = normalized_embds_generate(train_ben_loader, gnn_model, device, loss_require=False)
    print('[+] Train ben embds generated, len:{}'.format(len(train_ben_embds)))
    
    for iter_index in range(max_iters):
        print('iter:', iter_index, flush=True)
        # 计算 u1,u2,u3
        u1 = project_box(b_new + z1 / rho1)
        u2 = project_shifted_Lp_ball(b_new + z2 / rho2, projection_lp)
        u3 = project_positive(-torch.norm(b_new - b_ori, p=2) ** 2 + k_bits - z3 / rho3)
        # 根据loss，更新b（trigger）
        b_new, loss = update_mask_embd_dist(train_mal_loader, ben_embds=train_ben_embds, gnn_model=gnn_model, 
                                  u1=u1,u2=u2,u3=u3, z1=z1, z2=z2, z3=z3, k=k_bits,
                                  rho1=rho1, rho2=rho2, rho3=rho3, lambda_mal_loss=lambda_mal_loss, attack_node_type=attack_node_type,
                                  current_mask=b_new, ori_mask=b_ori, mask_range=[268, 492], device=device)
        
        if True in torch.isnan(b_new) or True in torch.isinf(b_new):
            print('nan or inf in mask')
            return -1

        z1 = z1 + rho1 * (b_new - u1)
        z2 = z2 + rho2 * (b_new - u2)
        z3 = z3 + rho3 * (torch.norm(b_new - b_ori, p=2) ** 2 - k_bits + u3)

        rho1 = min(rho_fact * rho1, max_rho1)
        rho2 = min(rho_fact * rho2, max_rho2)
        rho3 = min(rho_fact * rho3, max_rho3)

        temp1 = (torch.norm(b_new - u1)) / max(torch.norm(b_new), 2.2204e-16)
        temp2 = (torch.norm(b_new - u2)) / max(torch.norm(b_new), 2.2204e-16)
        print('Stop_threshold check', temp1.item(), temp2.item())
        
        if max(temp1, temp2) <= stop_threshold and iter_index > 100:
            print('END iter: %d, stop_threshold: %.6f, b_loss: %.4f' % (iter_index, max(temp1, temp2), loss))
            break
        
        if (iter_index+1) % eval_iter == 0:
            real_mask = torch.zeros(b_new.shape).to(device)
            real_mask[b_new > 0.5] = 1.0
            real_mask[b_new < 0.5] = 0.0
            if attack_node_type == 'all':
                trigger_model = Trigger_all_node_Model(trigger_range=[268, 492], trigger=real_mask)
            else:
                trigger_model = Trigger_Model(trigger_range=[268, 492], trigger=real_mask)
            evaluate_mask(train_mal_loader, gnn_model, trigger_model=trigger_model, device=device)
            evaluate_mask(test_mal_loader, gnn_model, trigger_model=trigger_model, device=device)
            torch.save(real_mask, '{}/mask_iter_{}.pt'.format(mask_save_dir, iter_index + 1))
        
    # 直接按照0.5 作为threshold作为标准来划分,映射回0，1
    mask_result = torch.zeros(b_new.shape)
    mask_result[b_new > 0.5] = 1.0
    mask_result[b_new < 0.5] = 0.0

    n_bit = torch.norm(b_ori - b_new)
    print('[+] End mask generation')
    print('[+] Mask bit num', n_bit)

    return mask_result

    
    

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
    aux_ben_subgraph_loader = DataLoader(aux_ben_subgraph_list, batch_size=1, shuffle=False)
    aux_mal_subgraph_loader = DataLoader(aux_mal_subgraph_list, batch_size=1, shuffle=False)
    val_ben_subgraph_loader = DataLoader(val_ben_subgraph_list, batch_size=1, shuffle=False)
    val_mal_subgraph_loader = DataLoader(val_mal_subgraph_list, batch_size=1, shuffle=False)
    print('[+] Done Process Dataset, aux_apk:{},aux_ben_subgraph:{},aux_mal_subgraph:{}'.format(len(aux_apk_list),len(aux_ben_subgraph_list), len(aux_mal_subgraph_list)))
    print('[+] Done Process Dataset, val_apk:{},val_ben_subgraph:{},val_mal_subgraph:{}'.format(len(val_apk_list),len(val_ben_subgraph_list), len(val_mal_subgraph_list)))
    print('[+] Load Dataset Done')  
    ##############################################################################################
    # mask_obtain = attack_origin(mal_loader=aux_mal_subgraph_loader, gnn_model=msdroid_model,attack_args=attack_args)
    mask_obtain = attack_vec_distance(train_mal_loader=aux_mal_subgraph_loader, 
                                      train_ben_loader=aux_ben_subgraph_loader,
                                      test_mal_loader=val_mal_subgraph_loader, test_ben_loader=val_ben_subgraph_loader,
                                      gnn_model=msdroid_model, attack_args=attack_args, 
                                      mask_save_dir=data_args.mask_save_dir, device=device)

    
if __name__ == "__main__":
    main()
    