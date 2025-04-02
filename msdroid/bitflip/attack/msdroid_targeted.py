import os
from tqdm import tqdm
import time
import random

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

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
    model_path = '/home/sample/lkc/MsDroid/src/training/Experiments/20250210-055931/models/last_epoch_200'
    layer_norm = True

class DataLoaderArguments:
    benign_path = '/home/sample/lkc/MsDroid/my_code/DataNew/Benign/HOP_2/TPL_True/dataset.pt'
    malware_path = '/home/sample/lkc/MsDroid/my_code/DataNew/Drebin/HOP_2/TPL_True/dataset.pt'
    aux_val_data_dir = 'target_attack_data'
    aux_num = 256
    val_num = 256
    seed = 0

class AttackArguments:
    lambda_malware_loss = 5.0
    target_bit = 10
    
    trigger_path = '/home/sample/lkc/MsDroid/my_code/mask/mask_iter_330.pt'
    trigger_range = [268, 492] # 0, 268 是permission
    
    absmax_topk = 20 # 每次查看前多少个absmax
    weight_topk = 20
    
    criterion_scale = 100
    

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
    self_def_criterion = My_Criterion(attack_args.criterion_scale)
    changed_bit = []
    ##############################################################################################
    # Load model
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
    trigger = nn.Parameter(torch.load(attack_args.trigger_path))
    trigger_model = Trigger_Model(attack_args.trigger_range, trigger=trigger)
    absmax_base = [2 ** i for i in range(16 - 1, -1, -1)]
    absmax_base[0] = -absmax_base[0]
    absmax_base = torch.tensor(absmax_base,dtype=torch.int16).cuda()
    weight_base = [2 ** i for i in range(8 - 1, -1, -1)]
    weight_base[0] = -weight_base[0]
    weight_base = torch.tensor(weight_base,dtype=torch.int16).cuda()
    print(trigger)
    print('[+] Done Load Trigger')
    
    # precision: 0.9377289377289377 recall: 1.0 accuracy: 0.966796875
    # precision: 0.920863309352518 recall: 1.0 accuracy: 0.95703125
    # precision: 0.9447619047619048 recall: 0.992 accuracy: 0.967
    # precision: 0.9767441860465116 recall: 0.984375 accuracy: 0.98046875
    # aux_data = torch.load('/home/sample/lkc/MsDroid/my_code/attack_data/aux_data.pt')
    # small_val_data = torch.load('/home/sample/lkc/MsDroid/my_code/attack_data/small_val.pt')
    # aux1_data = torch.load('/home/sample/lkc/MsDroid/my_code/aux_val_data/aux.pt')
    # small_val1_data = torch.load('/home/sample/lkc/MsDroid/my_code/aux_val_data/val.pt')
    # aux1_data = [Data(data=apk_graphs) for apk_graphs in aux1_data]
    # small_val1_data = [Data(data=apk_graphs) for apk_graphs in small_val1_data]
    
    # datasets = [aux1_data, aux_data, small_val_data, small_val1_data]
    # for dataset in datasets:
    #     norm_opcode_dataset(dataset)
    #     dataset_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    #     precision, recall, accuracy = evaluate_model1(loader=dataset_loader, model=origin_model, dev=device)
    # exit()
    ##############################################################################################
    # Split Dataset 把所有benign malware subgraph分开, aux和val dataset
    # 从原来的数据中读取，然后存成一个list: aux.pt, val.pt,整个是一个list，list中的每一个元素是一个list，包含一个apk的所有子图
    # 用来检测攻击效果
    aux_apk_list, val_apk_list = fetch_data(data_args.benign_path, data_args.malware_path, 
                                                        aux_num=data_args.aux_num, val_num=data_args.val_num,
                                                        save_dir=data_args.aux_val_data_dir)
    aux_subgraph_list = [subgraph for aux_apk in aux_apk_list for subgraph in aux_apk] # 把子list的所有元素合并为一个list
    ben_path = os.path.join(data_args.aux_val_data_dir, 'ben.pt')
    mal_path = os.path.join(data_args.aux_val_data_dir, 'mal.pt')
    aux_ben_subgraph_list, aux_mal_subgraph_list = ben_mal_subgraph_list_fetch(model, aux_subgraph_list, device, ben_path, mal_path)
    # ben 只用一部分，减少计算
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
        print(f'[+] ext_epoch:{ext_iter},trigger malware loss:{trigger_malware_loss} malware_loss:{malware_loss}, benign_loss:{benign_loss}')
        print(f'[+] ext_epoch:{ext_iter},t_m_acc:{ trigger_correct / float(len(aux_mal_subgraph_loader))} m_acc:{mal_correct / float(len(aux_mal_subgraph_loader))}, b_m_acc:{ben_correct / float(len(aux_ben_subgraph_loader))}')        
        
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
            sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)[: attack_args.absmax_topk]
            # attack info； key value: key: layer name, value: list，list的元素是absmax的(idx, grad)组成的元组
            attack_info={}
            for info in sorted_grad:
                layer, idx = info[0].split('@')
                if layer not in attack_info:
                    attack_info[layer] = []
                attack_info[layer].append((int(idx),info[1]))
            print('absmax attack_info')
            print(attack_info)
            # Step3 对grad最大的若干个absmax进行尝试 
            loss_each_trial = {}
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
                            loss_each_trial[bit_name] = total_loss
                            
                            layer.absmax[idx] = now_absmax.clone()
                        print('Single absmax time used:{:.2f}'.format(time.time() - single_absmax_start_time))
            print('loss_each_trial')
            print(loss_each_trial)
            # valid absmax
            best_bit, min_loss = min(loss_each_trial.items(), key=lambda item: item[1])
            skip_weight = False
            if min_loss < now_loss:
                skip_weight = True
        print('absmax finish, skip or not:{}'.format(skip_weight),flush=True)
        skip_weight = False
        loss_each_trial = {}
        # check weight
        with torch.no_grad():
            if not skip_weight:
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

                sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)[: attack_args.absmax_topk]

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
                            
                                trigger_malware_loss, _ = network_process_data(aux_mal_subgraph_loader, model=model, criterion=self_def_criterion,
                                                                           device=device, trigger_model=trigger_model, grad_need=False)
                                malware_loss, _ = network_process_data(aux_mal_subgraph_loader, model=model, criterion=self_def_criterion, 
                                                                    device=device, trigger_model=None, grad_need=False)
                                benign_loss, _ = network_process_data(aux_ben_subgraph_loader, model=model, criterion=self_def_criterion, 
                                                                    device=device, trigger_model=None, grad_need=False)
                                total_loss = benign_loss + malware_loss - attack_args.lambda_malware_loss * trigger_malware_loss
                                
                                best_bit  = f'{name}@{idx}@{i}@weight'
                                loss_each_trial[best_bit] = total_loss.data 
                                    
                                layer.w_int.data = layer.w_int.data.view(-1)
                                layer.w_int[idx] = now_weight.clone()
                                
                        layer.w_int.data = layer.w_int.data.view(ori_shape)
        
        best_bit, best_loss = min(loss_each_trial.items(), key=lambda item: item[1])
        if best_loss > now_loss:
            print(f'Loss increase, End')
            break
        
        # Step 5: 修改原来模型         
        with torch.no_grad():
            # select
            best_bit, best_loss = min(loss_each_trial.items(), key=lambda item: item[1])
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
