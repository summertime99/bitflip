import os
from tqdm import tqdm
import time
import random

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

from utils.utils import load_model, load_int8_model, set_random_seed
from utils.quant_model import find_all_bnbLinear, replace_with_myLinear, my_8bit_linear
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
        self.aux_val_data_dir = 'target_attack_data' # 用于攻击和验证的数据集
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


class My_Criterion(nn.Module):
    def __init__(self, scale):
        super(My_Criterion, self).__init__()
        self.scale = scale
    def forward(self, pred, label):
        pred = torch.exp(pred.squeeze() / self.scale)
        label_vec = torch.zeros(len(pred)).to(pred.device)
        label_vec[label] = 1
        label_vec = label_vec - 0.5
        return torch.dot(pred, label_vec)

def main(args):
    model_args = ModelArguments()
    data_args = DataLoaderArguments()
    attack_args = AttackArguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device('cuda')
    set_random_seed(data_args.seed)
    
    # TODO 考虑这个loss的合理性
    self_def_criterion = My_Criterion(attack_args.criterion_scale)
    self_def_criterion = torch.nn.CrossEntropyLoss()
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
    origin_model.to(device)
    
    clean_model = load_int8_model('', model_args.layer_norm)
    clean_model.load_state_dict(origin_model.state_dict())
    clean_model.to(device)
    print(clean_model)
    print('[+] Done Load Clean Model')
    
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
    ##############################################################################################
    # 加载trigger
    trigger_tensor = torch.zeros(224)
    trigger_tensor[40:43] = 1
    trigger = nn.Parameter(trigger_tensor)
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
    aux_ben_subgraph_path = os.path.join(data_args.aux_val_data_dir, 'ben_subgraph.pt')
    aux_mal_subgraph_path = os.path.join(data_args.aux_val_data_dir, 'mal_subgraph.pt')
    # aux中mal和benign的子图
    aux_ben_subgraph_list, aux_mal_subgraph_list = ben_mal_subgraph_list_fetch(model, aux_subgraph_list, device, aux_ben_subgraph_path, aux_mal_subgraph_path)
    # ben只用mal子图数量的两倍，来减少大量计算
    ben_index = random.sample(range(len(aux_ben_subgraph_list)), 2 * len(aux_mal_subgraph_list))
    aux_ben_subgraph_list = [aux_ben_subgraph_list[i] for i in ben_index]
    
    aux_ben_subgraph_loader = DataLoader(aux_ben_subgraph_list, batch_size=1, shuffle=False)
    aux_mal_subgraph_loader = DataLoader(aux_mal_subgraph_list, batch_size=1, shuffle=False)
    print('[+] Done Process Dataset, aux_apk:{},aux_ben_subgraph:{},aux_mal_subgraph:{},val_apk_num:{}'.format(len(aux_apk_list),len(aux_ben_subgraph_loader), len(aux_mal_subgraph_loader) ,len(val_apk_list)))
    ##############################################################################################    
    evaluate_target_attack(aux_apk_list, model, trigger_model, device)
    print('[+] Attack effect before bitflip')
    ##############################################################################################
    # 攻击模型
    for ext_iter in range(attack_args.target_bit):
        torch.cuda.empty_cache()
        model.zero_grad()
        # Step1 计算每一个位置的grad，找grad最大的元素
        trigger_malware_loss, trigger_correct = network_process_data(aux_mal_subgraph_loader, model=model, criterion=self_def_criterion, device=device,
                                            trigger_model=trigger_model, grad_need=True)
        malware_loss, mal_correct = network_process_data(aux_mal_subgraph_loader, model=model, criterion=self_def_criterion, device=device,
                                            trigger_model=None, grad_need=True)
        benign_loss, ben_correct = network_process_data(aux_ben_subgraph_loader, model=model, criterion=self_def_criterion, device=device,
                                            trigger_model=None, grad_need=True)
        
        # 希望加上trigger之后malware分类错误，同时不加trigger分类正确
        total_loss = benign_loss + malware_loss - attack_args.lambda_malware_loss * trigger_malware_loss 
        total_loss.backward()
        
        print('[+] ext_epoch:{}. Subgraph Level Data, trigger malware loss:{} malware loss:{}, benign loss:{}'.format(
            ext_iter, trigger_malware_loss, malware_loss, benign_loss))
        mal_subgraph_len = float(len(aux_mal_subgraph_loader))
        ben_subgraph_len = float(len(aux_ben_subgraph_loader))
        print('[+] ext_epoch:{}. Subgraph Level Data, trigger malware acc:{} malware acc:{}, benign acc:{}'.format(
            ext_iter, trigger_correct / mal_subgraph_len, mal_correct / mal_subgraph_len, ben_correct / ben_subgraph_len))     
        
        now_loss = total_loss
        
        # check absmax
        with torch.no_grad():
            # Step 2：找到grad最大的若干个absmax
            layers = {} # 记录每一层的absmax从小到大的顺序
            for name, layer in model.named_modules():
                if check_absmax_available(name, layer):
                    grad = layer.absmax.grad.data
                    grad = grad.view(-1) # 变成一维的数据
                    grad_abs = grad.abs()
                    grad_abs_sorted, grad_indices = torch.sort(grad_abs)
                    assert grad_indices.dtype == torch.int64
                    layers[name] = {'values': grad_abs_sorted.tolist(), 'indices':grad_indices.tolist()}
                
            all_grad = {} # key:absmax的位置; value:absmax的grad的大小
            for layer in layers:
                for i, idx in enumerate(layers[layer]['indices']):
                    all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

            # 所有absmax的topk
            sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)[: attack_args.topk_absmax]
            # attack info； key value: key: layer name, value: list，list的元素是absmax的(idx, grad)组成的元组
            attack_info={}
            for info in sorted_grad:
                layer, idx = info[0].split('@')
                if layer not in attack_info:
                    attack_info[layer] = []
                attack_info[layer].append((int(idx),info[1]))
            # print('absmax attack_info')
            # print(attack_info)
            # Step3 对grad最大的若干个absmax进行尝试 
            loss_absmax_trial = {}
            for name, layer in model.named_modules():
                if name in attack_info:
                    ori_absmax = layer.absmax.detach().clone()
                    for idx, grad in attack_info[name]:
                        single_absmax_start_time = time.time()
                        now_absmax = ori_absmax[idx]
                        # now_absmax.view(torch.int16) 把now_absmax当作字符串，转换为int
                        bits = torch.tensor([int(b) for b in Bits(int=int(now_absmax.view(torch.int16)),length=16).bin]).cuda()
                        for i in range(16):
                            old_bit = bits[i].clone()
                            if old_bit == 0:
                                new_bit = 1
                            else:
                                new_bit = 0
                            bits[i] = new_bit
                            new_absmax = bits * absmax_base
                            new_absmax = torch.sum(new_absmax, dim=-1).type(torch.int16).to(now_absmax.device).view(torch.float16)

                            if torch.isnan(new_absmax).any() or torch.isinf(new_absmax).any():
                                bits[i] = old_bit
                                continue
                            
                            if (new_absmax-now_absmax)*grad > 0:
                                bits[i] = old_bit
                                continue
                            
                            layer.absmax[idx] = new_absmax.clone()
                            bits[i] = old_bit
                            
                            trigger_malware_loss, _ = network_process_data(aux_mal_subgraph_loader, model=model, criterion=self_def_criterion,
                                                                           device=device, trigger_model=trigger_model, grad_need=False)
                            malware_loss, _ = network_process_data(aux_mal_subgraph_loader, model=model, criterion=self_def_criterion, 
                                                                   device=device, trigger_model=None, grad_need=False)
                            benign_loss, _ = network_process_data(aux_ben_subgraph_loader, model=model, criterion=self_def_criterion, 
                                                                  device=device, trigger_model=None, grad_need=False)
                            total_loss = benign_loss + malware_loss - attack_args.lambda_malware_loss * trigger_malware_loss
                            bit_name = f'{name}@{idx}@{i}@absmax'
                            loss_absmax_trial[bit_name] = total_loss
                            
                            layer.absmax[idx] = now_absmax.clone()
                        print('Single absmax time used:{:.2f}'.format(time.time() - single_absmax_start_time))
            # print('loss_absmax_trial')
            # print(loss_absmax_trial)
            # valid absmax
            best_bit, best_loss = min(loss_absmax_trial.items(), key=lambda item: item[1])
            skip_weight = False
            if best_loss < now_loss:
                skip_weight = True
            print('Check filp absmax finish. Flip absmax or not:{}'.format(skip_weight),flush=True)
        # check weight
        with torch.no_grad():
            if not skip_weight:
                loss_weight_trial = {}
                layers = {}
                for name, layer in model.named_modules():
                    if check_absmax_available(name, layer):
                        grad = layer.w_int.grad.data
                        grad = grad.view(-1)
                        grad_abs = grad.abs() 
                        grad_abs_sorted, grad_indices = torch.sort(grad_abs)
                        assert grad_indices.dtype == torch.int64
                        layers[name] = {'values': grad_abs_sorted.tolist(), 'indices':grad_indices.tolist()}
                                                                        
                all_grad = {}
                for layer in layers:
                    for i, idx in enumerate(layers[layer]['indices']):
                        all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

                sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)[: attack_args.topk_weight]

                attack_info={}
                for info in sorted_grad:
                    layer, idx = info[0].split('@')
                    if layer not in attack_info:
                        attack_info[layer] = []
                    attack_info[layer].append((int(idx),info[1]))

                for name, layer in model.named_modules():
                    if name in attack_info:
                        ori_shape = layer.w_int.shape
                        layer.w_int.data = layer.w_int.data.view(-1)
                        ori_weight = layer.w_int.detach().clone()
                        for idx, grad in attack_info[name]:
                            now_weight = ori_weight[idx]
                            bits = torch.tensor([int(b) for b in Bits(int=int(now_weight.type(torch.int8)),length=8).bin]).cuda()
                            for i in range(8):
                                old_bit = bits[i].clone()
                                if old_bit == 0:
                                    new_bit = 1
                                else:
                                    new_bit = 0
                                bits[i] = new_bit
                                new_weight = bits * weight_base
                                new_weight = torch.sum(new_weight, dim=-1).to(now_weight.device).type(torch.float16)
                                
                                if (new_weight-now_weight)*grad > 0:
                                    bits[i] = old_bit
                                    continue
                                
                                bits[i] = old_bit
                                
                                layer.w_int[idx] = new_weight.clone()
                                layer.w_int.data = layer.w_int.data.view(ori_shape)
                            
                                trigger_malware_loss, _ = network_process_data(aux_mal_subgraph_loader, model=model, 
                                                                               criterion=self_def_criterion, device=device,
                                                                               trigger_model=trigger_model, grad_need=False)
                                malware_loss, _ = network_process_data(aux_mal_subgraph_loader, model=model, criterion=self_def_criterion, 
                                                                    device=device, trigger_model=None, grad_need=False)
                                benign_loss, _ = network_process_data(aux_ben_subgraph_loader, model=model, criterion=self_def_criterion, 
                                                                    device=device, trigger_model=None, grad_need=False)
                                total_loss = benign_loss + malware_loss - attack_args.lambda_malware_loss * trigger_malware_loss
                                
                                best_bit  = f'{name}@{idx}@{i}@weight'
                                loss_weight_trial[best_bit] = total_loss.data 
                                    
                                layer.w_int.data = layer.w_int.data.view(-1)
                                layer.w_int[idx] = now_weight.clone()
                                
                        layer.w_int.data = layer.w_int.data.view(ori_shape)
        
                best_bit, best_loss = min(loss_weight_trial.items(), key=lambda item: item[1])
                if best_loss > now_loss:
                    print(f'Loss increase, End')
                    break
        
        # Step 5: 修改原来模型         
        with torch.no_grad():
            print(f'[+] change {best_bit}, loss: {best_loss}')
            # '{name}@{idx}@{i}@absmax', idx 是absmax在其中的位置，i是bit的位置
            layer_name, idx, i, bit_type = best_bit.split('@')
            idx, i = int(idx), int(i)
            for name, layer in model.named_modules():
                if isinstance(layer, my_8bit_linear) and layer_name == name:
                    if bit_type == 'absmax':
                        now_absmax = layer.absmax[idx]
                        bits = torch.tensor([int(b) for b in Bits(int=int(now_absmax.view(torch.int16)),length=16).bin]).cuda()
                        old_bit = bits[i]
                        if old_bit == 0:
                            new_bit = 1
                        else:
                            new_bit = 0
                        bits[i] = new_bit
                        new_absmax = bits * absmax_base
                        new_absmax = torch.sum(new_absmax, dim=-1).type(torch.int16).to(now_absmax.device).view(torch.float16)
                        layer.absmax[idx] = new_absmax.clone()
                    elif bit_type == 'weight':
                        ori_shape = layer.w_int.shape
                        layer.w_int.data = layer.w_int.data.view(-1)
                        now_weight = layer.w_int[idx]
                        bits = torch.tensor([int(b) for b in Bits(int=int(now_weight.type(torch.int8)),length=8).bin]).cuda()
                        old_bit = bits[i].clone()
                        if old_bit == 0:
                            new_bit = 1
                        else:
                            new_bit = 0
                        bits[i] = new_bit
                        new_weight = bits * weight_base
                        new_weight = torch.sum(new_weight, dim=-1).to(now_weight.device).type(torch.float16)
                        layer.w_int[idx] = new_weight.clone()
                        layer.w_int.data = layer.w_int.data.view(ori_shape)
                    else:
                        raise NotImplementedError
        # Step6: 评价模型
        if best_bit in changed_bit:
            changed_bit.remove(best_bit)
            print(f'[-] Revoke Flip {best_bit}')
        else:
            changed_bit.append(best_bit)
        
        evaluate_target_attack(aux_apk_list, model, trigger_model, device)
        nbit = len(changed_bit)
        print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)
        if len(changed_bit) >= attack_args.target_bit:
            break
                
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    
    args = parser.parse_args()
    
    main(args)
