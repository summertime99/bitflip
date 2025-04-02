import os
import numpy as np
import torch
from tqdm import tqdm
from bitstring import Bits

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))
utils_path=os.path.abspath('../')
import sys
sys.path.append(utils_path)

from utils.quant_model import find_all_bnbLinear, replace_with_myLinear, my_8bit_linear
from utils.models import Robust_AMD, Robust_AMD_INT8, Trigger_Model
from utils.load_data import load_data_robust_amd_targeted
from utils.metrics import robust_amd_acc, robust_amd_asr, robust_amd_loss_cal

benign_path = '../data/benign.npy'
malware_path = '../data/malware.npy'
vae_path = '../model/vae_model_f32.pth'
mlp_path = '../model/mlp_model_f32.pth'

def check_viable_module(name, module):
    if isinstance(module, my_8bit_linear) and 'decoder' not in name:
        return True
    return False

class TriggerArguments:
    def __init__(self):
        self.permission_range = [0,147]
        self.permission_vec_len = 147
        self.trigger_path = ''
        
class DataLoaderArguments:
    def __init__(self):
        self.aux_num = 256
        self.aux_mal_num = 256
        self.seed = 0
        self.batch_size = 64
        self.num_workers = 0
        self.split_ratio = 0.5
    
class AttackArguments:
    def __init__(self):
        self.target_class = 0
        self.orign_class = 1
        self.topk = 40 # for absmax (or 'Scale Factor')
        self.topk2 = 100 # for weight
        self.gamma = 1.
        self.target_bit = 50

# 0 benign, 1 malware

def main():
    dataargs = DataLoaderArguments()
    attackrargs = AttackArguments()
    triggerargs = TriggerArguments()
    print(dataargs.__dict__)
    print(attackrargs.__dict__)
    print(triggerargs.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fp32_model = Robust_AMD(vae_path=vae_path, mlp_path=mlp_path)
    
    model = Robust_AMD_INT8()
    model.load_state_dict(fp32_model.state_dict())
    model.to(0)
    
    clean_model = Robust_AMD_INT8()
    clean_model.load_state_dict(fp32_model.state_dict())
    clean_model.to(0)
    
    print('[+] Done Load Model')
    
    ##############################################################################################
    # Split dataset
    crossentropyloss = torch.nn.CrossEntropyLoss()
    mseloss = torch.nn.MSELoss()
    val_loader, aux_loader, small_val_loader, aux_mal_loader = load_data_robust_amd_targeted(benign_path, malware_path, 
                                                                    aux_num=dataargs.aux_num, aux_mal_num=dataargs.aux_mal_num,
                                                                    batch_size=dataargs.batch_size, split_ratio=dataargs.split_ratio)  
    # Replace with self-def linear
    _, modules_to_convert = find_all_bnbLinear(model)
    model, has_been_replaced = replace_with_myLinear(model, modules_to_convert=modules_to_convert, use_our_BFA=True)
    if not has_been_replaced:
        print("[-] Can't find any bnb Linear!")
        exit(0)
    print(model)
    print('[+] Done Replace Model')
    ##############################################################################################
    trigger_model = Trigger_Model(triggerargs.permission_vec_len, triggerargs.permission_range)
    
    trigger = torch.zeros(triggerargs.permission_vec_len)
    trigger[50] = 1 # 50:24, 70: 60.203, 22:12
    # trigger[64] = 1 # 64:61.896
    # trigger[134] = 1 # 134:61.268
    # trigger = torch.load(triggerargs.trigger_path, weights_only=True)
    trigger = torch.nn.Parameter(trigger, requires_grad=False)
    trigger_model.trigger = trigger
    trigger_model.to(device)
    print(trigger)
    print(f'[+] done load Trigger')
    ##############################################################################################
    # asr Attack Success Rate
    print('========================Before Attack========================')
    acc1 = robust_amd_acc(model, small_val_loader, device)
    asr1 = robust_amd_asr(model, small_val_loader, trigger_model, ori_class=1, target_class=0, device=device)
    print('========================Start  Attack========================')
    
    topk = attackrargs.topk
    topk2 = attackrargs.topk2
    gamma = attackrargs.gamma
    changed_bit = set()
    
    base = [2 ** i for i in range(16 - 1, -1, -1)]
    base[0] = -base[0]
    base = torch.tensor(base,dtype=torch.int16).cuda()
    
    base2 = [2 ** i for i in range(8 - 1, -1, -1)]
    base2[0] = -base2[0]
    base2 = torch.tensor(base2,dtype=torch.float16).cuda()
    
    target_class = attackrargs.target_class
    ori_class = attackrargs.orign_class
    
    for ext_iter in tqdm(range(attackrargs.target_bit+10)):
        torch.cuda.empty_cache()
        model.zero_grad()
        total_loss = 0.
        # 计算bitflip的model，加trigger之后和目标分类之间的crossentropy；计算bitflip的model和clean model输出之间的mse loss
        loss_remain = robust_amd_loss_cal(model, aux_loader, mseloss, device, clean_model=clean_model, trigger_model=None, grad_need=True)
        loss_attack = robust_amd_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, grad_need=True)
        total_loss = (loss_remain + gamma * loss_attack) / (1 + gamma)
        
        layers = {}
        print(f'[+] ext_epoch {ext_iter}: loss {total_loss}',flush=True)
        now_loss = total_loss
        
        # check absmax
        with torch.no_grad():
            # 找到grad最大的若个点
            for name, layer in model.named_modules():
                if check_viable_module(name, layer):
                    grad = layer.absmax.grad.data
                    grad = grad.view(-1)
                    grad_abs = grad.abs()
                    if len(grad_abs) < topk:
                        layers[name] = {'values': grad_abs, 'indices':range(len(grad_abs))}
                    else:
                        values, indices = grad_abs.topk(topk)
                        layers[name] = {'values': values.tolist(), 'indices': indices.tolist()}
                
            all_grad = {}
            for layer in layers:
                for i, idx in enumerate(layers[layer]['indices']):
                    all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

            sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)

            atk_info={}
            for info in sorted_grad[:topk]:
                layer, idx = info[0].split('@')
                if not atk_info.__contains__(layer):
                    atk_info[layer] = []
                atk_info[layer].append((int(idx),info[1]))

            all_loss = {}
            for name, layer in model.named_modules():
                if check_viable_module(name, layer):
                    if atk_info.__contains__(name):
                        ori_absmax = layer.absmax.detach().clone()
                        
                        for idx, grad in atk_info[name]:
                            now_absmax = ori_absmax[idx]
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
                                
                                bits[i] = old_bit
                                
                                layer.absmax[idx] = new_absmax.clone()
                                
                                loss_remain = robust_amd_loss_cal(model, aux_loader, mseloss, device, clean_model=clean_model,
                                                                  trigger_model=None, grad_need=False)
                                loss_attack = robust_amd_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None,
                                                                  trigger_model=trigger_model, grad_need=False)
                                total_loss = (loss_remain + gamma * loss_attack) / (1 + gamma)

                                layer.absmax[idx] = now_absmax.clone()
                                
                                if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                                    continue
                                
                                best_bit = f'{name}@{idx}@{i}@absmax'
                                all_loss[best_bit] = total_loss.data
                                
        # valid absmax
        # modify to judge whether all_loss is empty
        if len(all_loss) == 0:
            skip = False
        else:
            best_bit = min(all_loss, key=all_loss.get)
            min_loss = all_loss[best_bit]
            skip = False
            if min_loss < now_loss:
                skip = True

        # check weight
        if not skip:
            with torch.no_grad():
                for name, layer in model.named_modules():
                    if check_viable_module(name, layer):
                        grad = layer.w_int.grad.data
                        grad = grad.view(-1)
                        grad_abs = grad.abs()
                        if len(grad_abs) < topk2:
                            layers[name] = {'values': grad_abs, 'indices':range(len(grad_abs))}
                        else:
                            values, indices = grad_abs.topk(topk)
                            layers[name] = {'values': values.tolist(), 'indices': indices.tolist()}
                all_grad = {}
                for layer in layers:
                    for i, idx in enumerate(layers[layer]['indices']):
                        # if baned_weight.__contains__(layer) and idx in baned_weight[layer]:
                        #     continue
                        all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

                sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)

                atk_info={}
                for info in sorted_grad[:topk2]:
                    layer, idx = info[0].split('@')
                    if not atk_info.__contains__(layer):
                        atk_info[layer] = []
                    atk_info[layer].append((int(idx),info[1]))

                for name, layer in model.named_modules():
                    if check_viable_module(name, layer):
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
                                    
                                    loss_remain = robust_amd_loss_cal(model, aux_loader, mseloss, device, clean_model=clean_model,
                                                                  trigger_model=None, grad_need=False)
                                    loss_attack = robust_amd_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None,
                                                                  trigger_model=trigger_model, grad_need=False)
                                    total_loss = (loss_remain + gamma * loss_attack) / (1 + gamma)

                                    layer.w_int.data = layer.w_int.data.view(-1)
                                    layer.w_int[idx] = now_weight.clone()
                                
                                    best_bit  = f'{name}@{idx}@{i}@weight'
                                    all_loss[best_bit] = total_loss.data
                            
                            layer.w_int.data = layer.w_int.data.view(ori_shape)
        
        with torch.no_grad():       
            # select
            best_bit = min(all_loss, key=all_loss.get)
            print(f'[+] change {best_bit}, loss: {all_loss[best_bit]}',flush=True)
            # '{name}@{idx}@{i}@absmax'
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
            acc1 = robust_amd_acc(model, small_val_loader, device)
            asr1 = robust_amd_asr(model, small_val_loader, trigger_model, ori_class, target_class, device)
            try:
                changed_bit.remove(best_bit)
                print(f'[-] Revoke Flip {best_bit}')
            except:
                changed_bit.add(best_bit)
                if len(changed_bit) >= attackrargs.target_bit:
                    print('===========================End opt===========================')
                    acc1 = robust_amd_acc(model, small_val_loader, device)
                    asr1 = robust_amd_asr(model, small_val_loader, trigger_model, ori_class, target_class, device)
                    nbit = len(changed_bit)
                    print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)
                    exit(0)
                
    ##############################################################################################
    # End opt
    print('===========================End opt===========================')
    acc1 = robust_amd_acc(model, small_val_loader, device)
    asr1 = robust_amd_asr(model, small_val_loader, trigger_model, ori_class, target_class, device)
    nbit = len(changed_bit)
    print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)

if __name__ == '__main__':
    import argparse

    main()
