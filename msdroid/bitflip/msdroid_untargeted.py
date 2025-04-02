import os

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

from utils.quant_model import find_all_bnbLinear, replace_with_myLinear, my_8bit_linear
from utils.model import GNNStack, GNNStack_INT8, GNNStack_INT8_2
from utils.utils import load_model, load_int8_model, load_data_msdroid, replace_linear_with_8bit
from utils.evaluate_model import evaluate_model, evaluate_model1

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import copy
import time

import pickle
import bitsandbytes.functional as F2
import bitsandbytes as bnb
from bitstring import Bits

def check_absmax_available(name, module):
    if not isinstance(module, my_8bit_linear):
        return False
    return True
    

class ModelArguments:
    def __init__(self):
        self.model_path = '/home/sample/lkc/torch_version/msdroid/model/best.pt'
        self.quantize_type = 1 # 1 表示只对GNN输出之后的Linear做量化
        self.layer_norm = False

class DataLoaderArguments:
    def __init__(self):   
        self.ben_path = '/home/sample/lkc/torch_version/msdroid/DataNew/Benign/HOP_2/TPL_True/dataset.pt'
        self.mal_path = '/home/sample/lkc/torch_version/msdroid/DataNew/Drebin/HOP_2/TPL_True/dataset.pt'
        self.aux_num = 256
        self.split_ratio = 1
        self.batch_size = 1
        self.shuffle = False
    
class AttackArguments:
    def __init__(self):
        self.topk_absmax = 10 # for absmax (or 'Scale Factor')
        self.topk_weight = 20 # for weight
        self.target_bit = 10

def main():
    model_args = ModelArguments()
    data_args = DataLoaderArguments()
    attack_args = AttackArguments()
    device = torch.device("cuda")
    print('[+] Done Load Args')
    print(model_args.__dict__)
    print(data_args.__dict__)
    print(attack_args.__dict__)
    

    origin_model = load_model(model_args.model_path, model_args.layer_norm)
    print('origin model:', origin_model)
    # 加载模型, robust amd 是直接state_dict转
    model = load_int8_model('', model_args.layer_norm, type=model_args.quantize_type)
    model.load_state_dict(origin_model.state_dict())
    model.to(device)
    clean_model = load_int8_model('', model_args.layer_norm, type=model_args.quantize_type)
    clean_model.load_state_dict(origin_model.state_dict())
    clean_model.to(device)
    print('8bit model:', model)
    print('[+] Done Load Model')
    _, modules_to_convert = find_all_bnbLinear(model)
    model, has_been_replaced = replace_with_myLinear(model, modules_to_convert=modules_to_convert, use_our_BFA=True)
    if not has_been_replaced:
        print("[-] Can't find any bnb Linear!")
        exit(0)
    print('[+] Done Replace Model')
    ##############################################################################################
    # Split dataset
    crossentropyloss = torch.nn.CrossEntropyLoss()
    mseloss = torch.nn.MSELoss()
    val_loader, aux_loader, small_val_loader = load_data_msdroid(data_args.ben_path, data_args.mal_path,
                                                                 aux_num=data_args.aux_num, batch_size=data_args.batch_size,
                                                                 split_ratio=data_args.split_ratio, shuffle=data_args.shuffle)
    print('[+] Done Load Data')    
    print('========================Before Attack========================')
    # INT8 model precision: 0.9690278176082593 recall: 0.6549718937778639 accuracy: 0.8170188020934289
    # origin model precision: 0.9757125652348454 recall: 0.9422368676100019 accuracy: 0.9593913549137429
    # INT8 bitsandbytes: precision: 0.9759920634920635 recall: 0.953479356464431 accuracy: 0.9650125993409575
    precision, recall, accuracy = evaluate_model1(loader=small_val_loader, model=origin_model, dev=device)
    precision, recall, accuracy = evaluate_model1(loader=small_val_loader, model=model, dev=device)    
    print('========================Start  Attack========================')
    # 攻击的一些参数
    topk_absmax = attack_args.topk_absmax
    topk_weight = attack_args.topk_weight
    changed_bit = set()
    base = [2 ** i for i in range(16 - 1, -1, -1)]
    base[0] = -base[0]
    base = torch.tensor(base,dtype=torch.int16).cuda()
    
    base2 = [2 ** i for i in range(8 - 1, -1, -1)]
    base2[0] = -base2[0]
    base2 = torch.tensor(base2,dtype=torch.float16).cuda()
    
    # 迭代攻击
    for ext_iter in tqdm(range(attack_args.target_bit+10)):
        torch.cuda.empty_cache()
        model.zero_grad()
        # loss和梯度计算
        total_loss = 0.
        from utils.evaluate_model import real_batch
        for batch_idx, aux_inputs in enumerate(aux_loader):
            real_aux_input, position = real_batch(aux_inputs)
            real_aux_input = real_aux_input.to(device)
            _, clean_logits = clean_model(real_aux_input) # 输出 [n, 256], [n,2] 这里的n是子图的数量，x的是一个大矩阵
            clean_aux_pred = clean_logits.topk(1).indices.squeeze(dim=-1)
            # compute output
            _, model_logits = model(real_aux_input)
            loss = -crossentropyloss(model_logits, clean_aux_pred)
            total_loss += loss.data
            loss.backward(retain_graph=True)
                
        print(f'[+] ext_epoch {ext_iter}: loss {total_loss/(batch_idx+1)}',flush=True)
        now_loss = total_loss
        
        # check absmax
        with torch.no_grad():
            # 记录每一层的grad绝对值最大的topk_absmax个值
            layers = {} 
            for name, layer in model.named_modules():
                if check_absmax_available(name, layer):
                    grad = layer.absmax.grad.data
                    grad = grad.view(-1) # 变成一维的数据
                    
                    grad_abs = grad.abs()
                    if len(grad_abs) < topk_absmax:
                        layers[name] = {'values': grad_abs, 'indices':range(len(grad_abs))}
                    else:
                        values, indices = grad_abs.topk(topk_absmax)
                        layers[name] = {'values': values.tolist(), 'indices': indices.tolist()}
                
            # 每一个key-value key:absmax的位置; value:absmax的grad的大小    
            all_grad = {}
            for layer in layers:
                for i, idx in enumerate(layers[layer]['indices']):
                    all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]
            
            # sorted grad 只找每一层的topk的topk
            sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)
            sorted_grad = sorted_grad[:topk_absmax]
            # attack info； key value: key: layer name, value: list，list的元素是absmax的(idx, grad)组成的元组
            atk_info={}
            for info in sorted_grad:
                layer, idx = info[0].split('@')
                if layer not in atk_info:
                    atk_info[layer] = []
                atk_info[layer].append((int(idx),info[1]))
            
            # 针对上面找到的若干个最大的absmax，尝试其中的每一个bit，并记录修改之后的loss
            all_loss = {}
            for name, layer in model.named_modules():
                if isinstance(layer, my_8bit_linear) and name in atk_info:
                    ori_absmax = layer.absmax.detach().clone()
                    for idx, grad in atk_info[name]:
                        now_absmax = ori_absmax[idx]
                        # now_absmax.view(torch.int16) 把now_absmax当作字符串，转换为int
                        bits = torch.tensor([int(b) for b in Bits(int=int(now_absmax.view(torch.int16)),length=16).bin]).cuda()
                        flag = True
                        changable = []
                        for i in range(16):
                            old_bit = bits[i].clone()
                            if old_bit == 0:
                                new_bit = 1
                            else:
                                new_bit = 0
                            bits[i] = new_bit
                            new_absmax = bits * base
                            new_absmax = torch.sum(new_absmax, dim=-1).type(torch.int16).to(now_absmax.device).view(torch.float16)
                            
                            if torch.isnan(new_absmax).any() or torch.isinf(new_absmax).any():
                                bits[i] = old_bit
                                continue
                            
                            if (new_absmax-now_absmax)*grad > 0:
                                bits[i] = old_bit
                                continue
                            
                            layer.absmax[idx] = new_absmax.clone()
                            bits[i] = old_bit
                            
                            total_loss = 0.
                            for batch_idx, aux_inputs in enumerate(aux_loader):
                                real_aux_input, _ = real_batch(aux_inputs)
                                real_aux_input = real_aux_input.to(device)
                                _, clean_logits = clean_model(real_aux_input)
                                clean_aux_pred = clean_logits.topk(1).indices.squeeze(dim=-1)
                                # compute output
                                _, model_logits = model(real_aux_input)
                                loss = -crossentropyloss(model_logits, clean_aux_pred)
                                total_loss += loss.data
                                
                                if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                                    print("nan inf exist in total_loss")
                                    break

                            layer.absmax[idx] = now_absmax.clone()
                            
                            bit_name = f'{name}@{idx}@{i}@absmax'
                            all_loss[bit_name] = total_loss
            
            print('all_loss')
            print(all_loss)
        
        # 在absmax所有尝试中选择loss最小的，如果比原先的loss小，那么就翻转bit，进入下一个循环
        # 否则检查所有的weight参数
        best_bit = min(all_loss, key=all_loss.get)
        min_loss = all_loss[best_bit]
        skip = False
        if min_loss < now_loss:
            skip = True

        # check weight
        if not skip:
            with torch.no_grad():
                for name, layer in model.named_modules():
                    if isinstance(layer, my_8bit_linear):
                        grad = layer.w_int.grad.data
                        grad = grad.view(-1)
                        
                        grad_abs = grad.abs()
                        now_topk = grad.abs().topk(topk_weight)
                        layers[name] = {'values': grad[now_topk.indices].tolist(), 'indices':now_topk.indices.tolist()}
                    
                all_grad = {}
                for layer in layers:
                    for i, idx in enumerate(layers[layer]['indices']):
                        all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

                sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)
                sorted_grad = sorted_grad[:topk_weight]
                
                atk_info={}
                for info in sorted_grad:
                    layer, idx = info[0].split('@')
                    if layer not in atk_info:
                        atk_info[layer] = []
                    atk_info[layer].append((int(idx),info[1]))

                all_loss = {}
                for name, layer in model.deit.encoder.named_modules():
                    if isinstance(layer, my_8bit_linear):
                        if atk_info.__contains__(name):
                            ori_shape = layer.w_int.shape
                            layer.w_int.data = layer.w_int.data.view(-1)
                            ori_weight = layer.w_int.detach().clone()
                            for idx, grad in atk_info[name]:
                                now_weight = ori_weight[idx]
                                bits = torch.tensor([int(b) for b in Bits(int=int(now_weight.type(torch.int8)),length=8).bin]).cuda()
                                flag = True
                                changable = []
                                for i in range(8):
                                    old_bit = bits[i].clone()
                                    if old_bit == 0:
                                        new_bit = 1
                                    else:
                                        new_bit = 0
                                    bits[i] = new_bit
                                    new_weight = bits * base2
                                    new_weight = torch.sum(new_weight, dim=-1).to(now_weight.device).type(torch.float16)
                                    
                                    if (new_weight-now_weight)*grad > 0:
                                        bits[i] = old_bit
                                        continue
                                    
                                    bits[i] = old_bit
                                    layer.w_int[idx] = new_weight.clone()
                                    layer.w_int.data = layer.w_int.data.view(ori_shape)
                                
                                    best_bit = f'{name}@{idx}@{i}@weight'
                                    
                                    total_loss = 0.
                                    for batch_idx, aux_inputs in enumerate(aux_loader):
                                        real_aux_input, _ = real_batch(aux_inputs)
                                        real_aux_input = real_aux_input.to(device)
                                        _, clean_logits = clean_model(real_aux_input)
                                        clean_aux_pred = clean_logits.topk(1).indices.squeeze(dim=-1)
                                        # compute output
                                        _, model_logits = model(real_aux_input)
                                        loss = -crossentropyloss(model_logits, clean_aux_pred)
                                        total_loss += loss.data
                                
                                    layer.w_int.data = layer.w_int.data.view(-1)
                                    layer.w_int[idx] = now_weight.clone()
                                
                                    best_bit  = f'{name}@{idx}@{i}@weight'
                                    all_loss[best_bit] = total_loss.data
                            
                            layer.w_int.data = layer.w_int.data.view(ori_shape)
                    
        with torch.no_grad():
            # select
            best_bit = min(all_loss, key=all_loss.get)
            print(f'[+] change {best_bit}, loss: {all_loss[best_bit]}',flush=True)
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
                        new_absmax = bits * base
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
                        new_weight = bits * base2
                        new_weight = torch.sum(new_weight, dim=-1).to(now_weight.device).type(torch.float16)
                        layer.w_int[idx] = new_weight.clone()
                        layer.w_int.data = layer.w_int.data.view(ori_shape)
                    else:
                        raise NotImplementedError
        
        
        if best_bit in changed_bit:
            # 之前修改过，这次改回去
            changed_bit.remove(best_bit)
            print(f'[-] Revoke Flip {best_bit}')
        else:
            # 当前bit之前没有修改过
            changed_bit.add(best_bit)
            precision, recall, acc1 = evaluate_model1(small_val_loader, model, device)
            nbit = len(changed_bit)
            print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)
            # 翻转足够数量bit/攻击后准确率已经足够低
            if len(changed_bit) >= attack_args.target_bit or acc1 < 0.2:
                break
                    
    ##############################################################################################
    # End opt
    print('===========================End opt===========================')
    precision, recall, accuracy = evaluate_model1(small_val_loader, model, device)
    nbit = len(changed_bit)
    print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)


if __name__ == '__main__':
    main()
