# GNN内部不考虑，GNN输出之后所有Linear层,只考虑absmax
import os
# GNN内部不考虑，GNN输出之后只考虑一个Linear层（有topk限制）
if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

from utils.quant_model import find_all_bnbLinear, replace_with_myLinear, my_8bit_linear
from utils.model import GNNStack, GNNStack_INT8
from utils.utils import load_model, load_int8_model, load_data_msdroid, replace_linear_with_8bit, norm_opcode_dataset
from utils.evaluate_model import evaluate_model, evaluate_model1

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import pickle
import bitsandbytes.functional as F2
import bitsandbytes as bnb
from bitstring import Bits

def check_absmax_available(name, module):
    if not isinstance(module, my_8bit_linear):
        return False
    return True

def validate_no_inf_nan(t0):
    assert torch.sum(torch.isnan(t0) + torch.isinf(t0)) == 0
    

class ModelArguments:
    model_path = '/home/sample/lkc/MsDroid/src/training/Experiments/20250210-055931/models/last_epoch_200'
    layer_norm = True

class DataLoaderArguments:
    benign_path = '/home/sample/lkc/MsDroid/my_code/DataNew/Benign/HOP_2/TPL_True/dataset.pt'
    malware_path = '/home/sample/lkc/MsDroid/my_code/DataNew/Drebin/HOP_2/TPL_True/dataset.pt'
    aux_num = 256
    split_ratio = 0.2
    batch_size = 1
    shuffle = False
    
class AttackArguments:
    topk_scaler = 40 # for absmax (or 'Scale Factor')
    topk_weight = 20 # for weight
    target_bit = 10

def main():
    model_args = ModelArguments()
    data_args = DataLoaderArguments()
    attack_args = AttackArguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device('cuda')
    
    origin_model = load_model(model_args.model_path, model_args.layer_norm)
    # origin_model.to(device)
    print('origin model:', origin_model)
    # 加载模型, robust amd 是直接state_dict转
    model = load_int8_model('', model_args.layer_norm)
    model.load_state_dict(origin_model.state_dict())
    model.to(device)
    clean_model = load_int8_model('', model_args.layer_norm)
    clean_model.load_state_dict(origin_model.state_dict())
    clean_model.to(device)
    print('8bit model:', model)
    print('[+] Done Load Model')
    # 报错记录：
    # self.ori_cb = self.ori_bnb_linear.weight.CB.clone().to(torch.float16).cuda() # if self.ori_bnb_linear.weight.CB is not None else None
    # AttributeError: 'NoneType' object has no attribute 'clone'
    # 设置新的int8 model，int8 layer设定has_fp16_weights=False    
    ##############################################################################################
    # Split dataset
    crossentropyloss = torch.nn.CrossEntropyLoss()
    mseloss = torch.nn.MSELoss()
    aux_data = torch.load('/home/sample/lkc/MsDroid/my_code/aux_data.pt')
    small_val_data = torch.load('/home/sample/lkc/MsDroid/my_code/small_val.pt')
    norm_opcode_dataset(aux_data)
    norm_opcode_dataset(small_val_data)
    aux_loader = DataLoader(aux_data, batch_size=data_args.batch_size, shuffle=False)
    small_val_loader = DataLoader(small_val_data, batch_size=data_args.batch_size, shuffle=False)
    print('[+] Done Load Data')
    # check_data_nan_inf(small_val_loader)
    #-----------------------Trojan Insertion----------------------------------------------------------------
    _, modules_to_convert = find_all_bnbLinear(model)
    model, has_been_replaced = replace_with_myLinear(model, modules_to_convert=modules_to_convert, use_our_BFA=True)
    if not has_been_replaced:
        print("[-] Can't find any bnb Linear!")
        exit(0)
    print('converted model')
    print(model)
    print('[+] Done Replace Model')
    print('========================Before Attack========================')
    # origin model precision: 0.9757125652348454 recall: 0.9422368676100019 accuracy: 0.9593913549137429
    # precision: 0.9447092469018112 recall: 0.991 accuracy: 0.9665
    # precision, recall, accuracy = evaluate_model1(loader=small_val_loader, model=origin_model, dev=device)
    # precision, recall, accuracy = evaluate_model1(loader=small_val_loader, model=clean_model, dev=device)
    # precision, recall, accuracy = evaluate_model1(loader=small_val_loader, model=model, dev=device)
    print('========================Start  Attack========================')
    changed_bit = set()
    topk_sacler = attack_args.topk_scaler
    topk_weight = attack_args.topk_weight
    
    base = [2 ** i for i in range(16 - 1, -1, -1)]
    base[0] = -base[0]
    base = torch.tensor(base,dtype=torch.int16).cuda()

    base2 = [2 ** i for i in range(8 - 1, -1, -1)]
    base2[0] = -base2[0]
    base2 = torch.tensor(base2,dtype=torch.float16).cuda()

    baned_absmax = {} # 指定禁用 部分层的部分weight
    baned_weight = {}
    allow_weight = {}
    
    model.eval()
    clean_model.eval()
    origin_model.eval()
    
    
    # aux_target 是clean model判断得到的数据的标签
    # loss 是评价clean model的预测和target model之间的相似程度的负值，希望这两个model之间的距离尽可能远，也就是loss尽可能小
    from utils.evaluate_model import real_batch
    clean_aux_pred_list = []
    with torch.no_grad():
        for batch_idx, aux_inputs in enumerate(aux_loader):
            real_aux_input, position = real_batch(aux_inputs)
            real_aux_input = real_aux_input.to(device)
            _, clean_logits = clean_model(real_aux_input) # 输出 [n, 256], [n,2] 这里的n是子图的数量，x的是一个大矩阵
            clean_aux_pred = torch.argmax(clean_logits, dim=1) 
            clean_aux_pred_list.append(clean_aux_pred)
    
    # 找对应的bit    
    for ext_iter in tqdm(range(attack_args.target_bit+10)):
        torch.cuda.empty_cache()
        model.zero_grad()
        total_loss = 0.
        # loss calculation
        for batch_idx, aux_inputs in enumerate(aux_loader):
            real_aux_input, position = real_batch(aux_inputs)
            real_aux_input = real_aux_input.to(device)
            # compute output
            _, model_logits = model(real_aux_input)
            validate_no_inf_nan(model_logits)
            loss = -crossentropyloss(model_logits, clean_aux_pred_list[batch_idx])
            validate_no_inf_nan(loss)
            total_loss += loss.data
            loss.backward(retain_graph=True)        
        print(f'[+] ext_epoch {ext_iter}: loss {total_loss/(batch_idx+1)}',flush=True)
        now_loss = total_loss
        # check absmax
        layers = {} # 记录每一层的最大的若干个absmax
        with torch.no_grad():
            for name, layer in model.named_modules():
                if check_absmax_available(name, layer):
                    grad = layer.absmax.grad.data
                    grad = grad.view(-1) # 变成一维的数据
                    grad_abs = grad.abs()
                    grad_abs_sorted, grad_indices = torch.sort(grad_abs)
                    assert grad_indices.dtype == torch.int64
                    layers[name] = {'values': grad_abs_sorted.tolist(), 'indices':grad_indices.tolist()}
            # 每一个key-value key:absmax的位置; value:absmax的grad的大小    
            all_grad = {}
            for layer in layers:
                for i, idx in enumerate(layers[layer]['indices']):
                    if baned_absmax.__contains__(layer) and idx in baned_absmax[layer]:
                        continue
                    all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]
            print("all_grad")
            print(all_grad)
            # sorted grad 只找每一层的topk的topk
            sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)[: topk_sacler]
            # attack info； key value: key: layer name, value: list，list的元素是absmax的(idx, grad)组成的元组
            atk_info={}
            for info in sorted_grad:
                layer, idx = info[0].split('@')
                if layer not in atk_info:
                    atk_info[layer] = []
                atk_info[layer].append((int(idx),info[1]))

            print("attack info")
            print(atk_info)

            all_loss = {}
            
            for name, layer in model.named_modules():
                if isinstance(layer, my_8bit_linear):
                    if name in atk_info:
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
                                print('change bit {}'.format(i), now_absmax, new_absmax)
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
                                    # compute output
                                    _, model_logits = model(real_aux_input)
                                    loss = -crossentropyloss(model_logits, clean_aux_pred_list[batch_idx])
                                    total_loss += loss.data
                                    
                                    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                                        print("nan inf exist in total_loss")
                                        break

                                layer.absmax[idx] = now_absmax.clone()
                                
                                bit_name = f'{name}@{idx}@{i}@absmax'
                                all_loss[bit_name] = total_loss
            
            print('all_loss')
            print(all_loss)
        # valid absmax
        best_bit = min(all_loss, key=all_loss.get)
        min_loss = all_loss[best_bit]
        skip = False
        if min_loss < now_loss:
            skip = True

        
                    
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
                        
                        start_ban = (idx // (4096//2)) * (4096//2) # Note that absmax is a float16 parameter
                        end_ban = start_ban + (4096//2)
                        end_ban = min(end_ban, len(layer.absmax))
                        for idx in range(start_ban, end_ban):
                            if name not in baned_absmax:
                                baned_absmax[name] = []
                            baned_absmax[name].append(idx)
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
                        
                        start_ban = (idx // (4096)) * 4096 # Note that weight is a int8 parameter
                        end_ban = start_ban + (4096)
                        end_ban = min(end_ban, len(layer.w_int))
                        for idx in range(start_ban, end_ban):
                            if name not in baned_weight:
                                baned_weight[name] = []
                            baned_weight[name].append(idx)
                        layer.w_int.data = layer.w_int.data.view(ori_shape)
                    else:
                        raise NotImplementedError
        
        try:
            changed_bit.remove(best_bit)
            print(f'[-] Revoke Flip {best_bit}')
            layer_name, idx, i, bit_type = best_bit.split('@')
            idx, i = int(idx), int(i)
            for name, layer in model.named_modules():
                if isinstance(layer, my_8bit_linear) and layer_name == name:
                    if bit_type == 'absmax':
                        start_ban = (idx // (4096//2)) * (4096//2) # Note that absmax is a float16 parameter
                        end_ban = start_ban + (4096//2)
                        end_ban = min(end_ban, len(layer.absmax))
                        for idx in range(start_ban, end_ban):
                            baned_absmax[name].remove(idx)
                    elif bit_type == 'weight':
                        start_ban = (idx // (4096)) * 4096 # Note that weight is a int8 parameter
                        end_ban = start_ban + (4096)
                        end_ban = min(end_ban, len(layer.w_int.view(-1)))
                        for idx in range(start_ban, end_ban):
                            baned_weight[name].remove(idx)
                    else:
                        raise NotImplementedError
        except:
            changed_bit.add(best_bit)
            if len(changed_bit) >= attack_args.target_bit:
                print('===========================End opt===========================')
                precision, recall, accuracy = evaluate_model1(small_val_loader, model, device)
                nbit = len(changed_bit)
                print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)
                exit(0)
            
            precision, recall, acc1 = evaluate_model1(small_val_loader, model, device)
            nbit = len(changed_bit)
            print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)
            if acc1 < 0.2:
                break
                    
    ##############################################################################################
    # End opt
    print('===========================End opt===========================')
    precision, recall, accuracy = evaluate_model1(small_val_loader, model, device)
    nbit = len(changed_bit)
    print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)



if __name__ == '__main__':
    main()
