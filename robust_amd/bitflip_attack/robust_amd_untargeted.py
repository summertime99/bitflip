import os
if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

from utils.quant_model import find_all_bnbLinear, replace_with_myLinear, my_8bit_linear
from utils.models import Robust_AMD, Robust_AMD_INT8
from utils.load_data import load_data_robust_amd
from utils.metrics import robust_amd_acc

import torch
from tqdm import tqdm
from bitstring import Bits

benign_path = '../data/benign.npy'
malware_path = '../data/malware.npy'
vae_path = '../model/vae_model_f32.pth'
mlp_path = '../model/mlp_model_f32.pth'

class DataLoaderArguments:
    def __init__(self):
        self.aux_num = 256
        self.seed = 0
        self.batch_size = 64
        self.shuffle = False
        self.num_workers = 0
        self.pin_memory = False
        self.split_ratio = 0.5
    
class AttackArguments:
    def __init__(self):
        self.topk = 20
        self.topk2 = 40 
        self.target_bit = 10

def check_viable_module(name, module):
    if isinstance(module, my_8bit_linear) and 'decoder' not in name:
        return True
    return False


def main():
    dataargs = DataLoaderArguments()
    attackrargs = AttackArguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device("cuda")
    
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
    val_loader, aux_loader, small_val_loader = load_data_robust_amd(benign_path, malware_path, 
                                                                    dataargs.aux_num, dataargs.batch_size, dataargs.split_ratio)
    #-----------------------Trojan Insertion----------------------------------------------------------------
    _, modules_to_convert = find_all_bnbLinear(model)
    model, has_been_replaced = replace_with_myLinear(model, modules_to_convert=modules_to_convert, use_our_BFA=True)
    if not has_been_replaced:
        print("[-] Can't find any bnb Linear!")
        exit(0)
    print(model)
    print('[+] Done Replace Model')
    
    print('========================Before Attack========================')
    acc1 = robust_amd_acc(model, small_val_loader, device)
    print('========================Start  Attack========================')
    
    topk = attackrargs.topk
    topk2 = attackrargs.topk2
    changed_bit = set()
    
    base = [2 ** i for i in range(16 - 1, -1, -1)]
    base[0] = -base[0]
    base = torch.tensor(base,dtype=torch.int16).cuda()
    
    base2 = [2 ** i for i in range(8 - 1, -1, -1)]
    base2[0] = -base2[0]
    base2 = torch.tensor(base2,dtype=torch.float16).cuda()
    
    for ext_iter in tqdm(range(attackrargs.target_bit+10)):
        torch.cuda.empty_cache()
        model.zero_grad()
        total_loss = 0.
        # 计算当前loss（modified）
        for batch_idx, (aux_inputs, _) in enumerate(aux_loader):
            aux_inputs = aux_inputs.to(device)
            # clean model 预测标签和攻击模型的输出之间的crossentropy
            clean_outputs, _, _, _ = clean_model(aux_inputs)
            clean_predicted = torch.argmax(clean_outputs, dim=1)

            attack_outputs, _, _, _ = model(aux_inputs)
            loss = -crossentropyloss(attack_outputs, clean_predicted)

            total_loss += loss.data
            loss.backward(retain_graph=True)
        
        print(f'[+] epoch {ext_iter}: loss {total_loss/(batch_idx+1)}',flush=True) # 此处loss取负
        now_loss = total_loss
        layers = {}
        # check absmax（scalar 缩放因子）
        with torch.no_grad():
            for name, layer in model.named_modules(): # model.deit.encoder.named_modules(): 改为model.named_modules(),改完可以遍历所有module
                if check_viable_module(name, layer): # decoder 不参与标签计算
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
                    # if baned_absmax.__contains__(layer) and idx in baned_absmax[layer]:
                    #     continue
                    all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

            sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)

            atk_info={}
            for info in sorted_grad[:topk]:
                layer, idx = info[0].split('@')
                if not atk_info.__contains__(layer):
                    atk_info[layer] = []
                atk_info[layer].append((int(idx),info[1]))

            all_loss = {}
            for name, layer in model.named_modules(): # model.deit.encoder.named_modules(): 改为model.named_modules()
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
                                
                                layer.absmax[idx] = new_absmax.clone()
                                bits[i] = old_bit
                                
                                total_loss = 0.
                                for batch_idx, (aux_inputs, _) in enumerate(aux_loader):
                                    aux_inputs = aux_inputs.to(device)
                                    # TODO 按照一开始计算loss的方式修改
                                    clean_outputs, _, _, _ = clean_model(aux_inputs)
                                    clean_predicted = torch.argmax(clean_outputs, dim=1)
                                    # compute output
                                    outputs, _, _, _ = model(aux_inputs)
                                    loss = -crossentropyloss(outputs, clean_predicted)
                                    total_loss += loss.data
                                    
                                    if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                                        break

                                layer.absmax[idx] = now_absmax.clone()
                                
                                if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                                    continue
                                
                                best_bit = f'{name}@{idx}@{i}@absmax'
                                all_loss[best_bit] = total_loss.data
        # check uppper part! finished

        # valid absmax
        best_bit = min(all_loss, key=all_loss.get)
        min_loss = all_loss[best_bit]
        skip = False
        if min_loss < now_loss:
            skip = True

        # check weight
        # bitflip weight
        if not skip:
            with torch.no_grad():
                for name, layer in model.deit.encoder.named_modules():
                    if check_viable_module(name, layer):
                        grad = layer.w_int.grad.data
                        grad = grad.view(-1)
                        
                        grad_abs = grad.abs()
                        # if baned_weight.__contains__(name):
                        #     for idx in baned_weight[name]:
                        #         grad_abs[idx] = -100.
                        
                        now_topk = grad.abs().topk(topk2)
                        layers[name] = {'values': grad[now_topk.indices].tolist(), 'indices':now_topk.indices.tolist()}
                    
                all_grad = {}
                for layer in layers:
                    for i, idx in enumerate(layers[layer]['indices']):
                        all_grad['@'.join([layer,str(idx)])] = layers[layer]['values'][i]

                sorted_grad = sorted(all_grad.items(), key = lambda x:abs(x[1]), reverse = True)

                atk_info={}
                for info in sorted_grad[:topk2]:
                    layer, idx = info[0].split('@')
                    if not atk_info.__contains__(layer):
                        atk_info[layer] = []
                    atk_info[layer].append((int(idx),info[1]))

                all_loss = {}
                for name, layer in model.deit.encoder.named_modules():
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
                                    
                                    total_loss = 0.
                                    for batch_idx, (aux_inputs, _) in enumerate(aux_loader):
                                        aux_inputs = aux_inputs.to(device)
                                        aux_targets = clean_model(aux_inputs).logits.topk(1).indices.squeeze(dim=-1)
                                        # compute output
                                        outputs = model(aux_inputs)
                                        logits = outputs.logits
                                        loss = -crossentropyloss(logits, aux_targets)
                                        
                                        total_loss += loss.data
                                
                                    layer.w_int.data = layer.w_int.data.view(-1)
                                    layer.w_int[idx] = now_weight.clone()
                                
                                    best_bit  = f'{name}@{idx}@{i}@weight'
                                    all_loss[best_bit] = total_loss.data
                            
                            layer.w_int.data = layer.w_int.data.view(ori_shape)
        # check trial bitflip weight finished
        
        # attack: flip bit
        with torch.no_grad():
            # select
            best_bit = min(all_loss, key=all_loss.get)
            print(f'[+] change {best_bit}, loss: {all_loss[best_bit]}',flush=True)
            # best_bit: '{name}@{idx}@{i}@absmax'
            layer_name, idx, i, bit_type = best_bit.split('@')
            idx, i = int(idx), int(i)
            for name, layer in model.named_modules():
                if check_viable_module(name, layer) and layer_name == name:
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
        
        try:
            changed_bit.remove(best_bit)
            print(f'[-] Revoke Flip {best_bit}')
        except:
            changed_bit.add(best_bit)
            if len(changed_bit) >= attackrargs.target_bit:
                print('===========================End opt===========================')
                acc1 = robust_amd_acc(model, small_val_loader, device)
                nbit = len(changed_bit)
                print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)
                exit(0)
            
            acc1 = robust_amd_acc(model, small_val_loader, device)
            nbit = len(changed_bit)
            print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)
            if acc1 < 0.2:
                break
                    
    ##############################################################################################
    # End opt
    print('===========================End opt===========================')
    acc1 = robust_amd_acc(model, small_val_loader, device)
    nbit = len(changed_bit)
    print(f'[+] Flip {nbit} bit: {changed_bit}',flush=True)



if __name__ == '__main__':
    main()
