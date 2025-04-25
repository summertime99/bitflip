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

from utils_admm_optim.quant_model import find_all_bnbLinear, replace_with_myLinear, my_8bit_linear, linear_tensor_dict
from utils_admm_optim.admm import set_model_project_save, set_model_origin, model_changed_bit
from utils_admm_optim.admm import set_model_grad_enable, set_model_grad_disable
from utils_admm_optim.admm import set_model_project_test, set_model_project_close
from utils_admm_optim.admm import update_u, update_z, grad_in_lagrange_normalize_loss
from utils_admm_optim.metrics import malconv_loss_cal

from utils.models import Malconv, Malconv_INT8, Trigger_Model, ADMM_Trigger_Model
from utils.load_data import load_data_malconv_targeted
from utils.metrics import malconv_acc, malconv_asr

benign_path = '../data/benign.pt'
malware_path = '../data/malware.pt'
model_path = '../model/best_bs=256_lr=0.001_wd=0.001.pt'

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
        self.target_class = 1
        self.orign_class = 0
        self.topk = 40 # for absmax (or 'Scale Factor')
        self.topk2 = 100 # for weight
        self.gamma = 1.
        self.target_bit = 40

# 0 benign, 1 malware

def main():
    dataargs = DataLoaderArguments()
    attackrargs = AttackArguments()
    triggerargs = TriggerArguments()
    print(dataargs.__dict__)
    print(attackrargs.__dict__)
    print(triggerargs.__dict__)
    device = torch.device("cuda")
    
    fp32_model = Malconv()
    fp32_model.load_state_dict(torch.load(model_path))
    #print('origin model:', fp32_model)
    fp32_model.to(device)
    
    model = Malconv_INT8()
    model.load_state_dict(fp32_model.state_dict())
    model.to(device)
    
    clean_model = Malconv_INT8()
    clean_model.load_state_dict(fp32_model.state_dict())
    clean_model.to(device)
    
    print('[+] Done Load Model')
    
    ##############################################################################################
    # Split dataset
    crossentropyloss = torch.nn.CrossEntropyLoss()
    mseloss = torch.nn.MSELoss()
    val_loader, aux_loader, small_val_loader, aux_mal_loader = load_data_malconv_targeted(benign_path, malware_path, 
                                                                    aux_num=dataargs.aux_num, aux_mal_num=dataargs.aux_mal_num,
                                                                    batch_size=dataargs.batch_size, split_ratio=dataargs.split_ratio)  
    ##############################################################################################
    trigger_model = ADMM_Trigger_Model(triggerargs.permission_vec_len, triggerargs.permission_range)
    # 这里可以手动设置一下trigger
    trigger_model.trigger.requires_grad = False
    
    # asr 19.782 loss:4.764
    # trigger_index_list = [0, 15, 16, 22, 28, 31, 32, 35, 106, 110, 115, 146]
    # asr 19.14 ; loss:4.99
    #trigger_index_list = [0, 16, 106, 110, 115]
    # asr 42.578 loss:438
    # trigger_index_list = [0, 2, 15, 16, 19, 22, 26, 32, 66, 87, 112, 115, 146]
    # asr 40.6 loss:4.55
    # trigger_index_list = [0, 2, 26, 87, 112]
    #for trigger_index in trigger_index_list:
    #    trigger_model.trigger[trigger_index] = 1
    
    trigger_model.to(device)
    #print(trigger_index_list)
    print(f'[+] done load Trigger')
    ##############################################################################################
    # asr Attack Success Rate
    print('========================Before Attack========================')
    print("FP32")
    malconv_acc(fp32_model, val_loader, device)
    print("INT8")
    malconv_acc(clean_model, val_loader, device)
    malconv_acc(model, val_loader, device)
    # Replace linear layer with self-def linear
    _, modules_to_convert = find_all_bnbLinear(model)
    model, has_been_replaced = replace_with_myLinear(model, modules_to_convert=modules_to_convert)
    
    if not has_been_replaced:
        print("[-] Can't find any bnb Linear!")
        exit(0)
    print('[+] Done Replace Model')
    model.to(device)
    print("myint8")
    malconv_acc(model, val_loader, device)
    malconv_asr(model, small_val_loader, trigger_model, ori_class=1, target_class=0, device=device)
    print('========================Start  Attack========================')
    
    # 初始化 ADMM 变量
    # 依次对不同的层进行bitflip attack
    # ['vae.encoder.0.weight_binary', 'vae.encoder.0.absmax_binary', 'vae.encoder.3.weight_binary', 'vae.encoder.3.absmax_binary', 'mlp.mlp.0.weight_binary', 'mlp.mlp.0.absmax_binary', 'mlp.mlp.6.weight_binary', 'mlp.mlp.6.absmax_binary', 'mlp.mlp.3.weight_binary', 'mlp.mlp.3.absmax_binary', 'vae.encoder.6.weight_binary', 'vae.encoder.6.absmax_binary', 'mlp.mlp.9.weight_binary', 'mlp.mlp.9.absmax_binary']
    target_class = attackrargs.target_class
    ori_class = attackrargs.orign_class
    k = attackrargs.target_bit
    
    trial_list = [['vae.encoder.0.absmax_binary'], ['vae.encoder.3.absmax_binary'], ['mlp.mlp.0.absmax_binary'], ['mlp.mlp.3.absmax_binary'], ['mlp.mlp.6.absmax_binary']]
    train_epoch = [100, 100, 100, 100, 100]
    
    # trial_list = [['vae.encoder.0.weight_binary', 'vae.encoder.0.absmax_binary'], ['vae.encoder.3.weight_binary', 'vae.encoder.3.absmax_binary']]
    # train_epoch = [100, 100]
    
    # trial_list = [['mlp.mlp.0.absmax_binary', 'mlp.mlp.0.weight_binary'], ['mlp.mlp.3.absmax_binary', 'mlp.mlp.3.weight_binary']]
    # train_epoch = [100, 100]
    # trial_list = [['mlp.mlp.0.absmax_binary'], ['mlp.mlp.0.weight_binary'], ['mlp.mlp.3.absmax_binary'], ['mlp.mlp.3.weight_binary']]
    # train_epoch = [100, 100, 100, 100]
    
    # trial_list = [['vae.encoder.0.absmax_binary'], ['vae.encoder.3.absmax_binary'], ['mlp.mlp.0.absmax_binary'], ['mlp.mlp.0.weight_binary'], ['mlp.mlp.3.absmax_binary'], ['mlp.mlp.3.weight_binary'], ['mlp.mlp.6.absmax_binary'], ['mlp.mlp.6.weight_binary'], ['mlp.mlp.9.weight_binary'],]
    # train_epoch = [100, 100, 100, 100, 100, 100, 100, 100, 100]
    
    
    
    max_epoch = 200
    
    flip_num = []
    
    for trial_index in range(len(trial_list)):
        print('[+] Start iter {}, {}'.format(trial_index, trial_list[trial_index]))
        para_name_in_model = set(trial_list[trial_index])
        
        b_ori = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
        
        z1 = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=False, if_0=True)
        z2 = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=False, if_0=True)
        z3 = [torch.zeros(1).to(device)]
        
        u1 = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
        u2 = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
        u3 = [torch.zeros(1).to(device)]

        rho1 = 0.05
        rho2 = 0.05
        rho3 = 0.0005
        
        max_rho1 = 5
        max_rho2 = 5
        max_rho3 = 0.5
        rho_fact = 1.01

        # 还剩余的bitflip的次数
        k_used = 0
        for i in flip_num:
            k_used += i
        k_available = k - k_used
        epoch_num = train_epoch[trial_index]
        
        print('k_available: {}, min epoch:{}'.format(k_available, epoch_num))
        
        if 'weight_binary' in trial_list[trial_index][0]:
            gamma = 1
        elif 'absmax_binary' in trial_list[trial_index][0]:
            gamma = 1
            
        
        b_real_dict = linear_tensor_dict(model, para_name_in_model, if_real=True, if_clone=False, if_0=False)
        b_real_list = list(b_real_dict.values())
        optimizer = torch.optim.SGD(b_real_list, lr=0.01, momentum=0.9)
        crossentropyloss = torch.nn.CrossEntropyLoss()
        mseloss = torch.nn.MSELoss()
        # 最初的loss_attack 只要比一开始进入的要好即可
        loss_attack_initial = 0
        loss_remain_initial = 0
        
        for iter_index in range(1,max_epoch + 1):
            optimizer.zero_grad()
            # 更新u
            b_new_dict = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
            update_u(b_ori_dict=b_ori, b_new_dict=b_new_dict, u1_dict=u1, u2_dict=u2, u3_list=u3, z1_dict=z1, z2_dict=z2, z3_list=z3,
                        rho1=rho1, rho2=rho2, rho3=rho3, k=k_available)
            
            # 计算loss 更新b
            model.eval() # 不需要管droppout
            set_model_grad_enable(model, para_name_in_model)
            # 攻击的loss remain是保持模型能力，attack是在有trigger下攻击模型
            # loss_remain = robust_amd_loss_cal(model, aux_loader, mseloss, device, clean_model=clean_model, trigger_model=None, require_grad=True)
            loss_remain = malconv_loss_cal(model, aux_loader, crossentropyloss, device, clean_model=None, trigger_model=None, require_grad=True)
            loss_attack = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=True)
            loss_attack = gamma * loss_attack
            if iter_index == 1:
                loss_attack_initial = loss_attack.item()
                loss_remain_initial = loss_remain.item()
            # 拉格朗日正则化项的loss
            # name 和 对应的grad组成dict
            lagrange_grad_dict = grad_in_lagrange_normalize_loss(b_ori_dict=b_ori, b_new_dict=b_new_dict, u1_dict=u1, u2_dict=u2, u3_list=u3,
                                                                z1_dict=z1, z2_dict=z2, z3_list=z3, rho1=rho1, rho2=rho2, rho3=rho3, k=k_available)
            total_loss = loss_remain + loss_attack
            
            if iter_index % 20 == 0:
                print('[+] Loss: loss remain right:{:.3f} loss attack:{:.3f}'.format(loss_remain.item(), loss_attack.item()))
                
            total_loss.backward()
            
            b_real_dict = linear_tensor_dict(model, para_name_in_model, if_real=True, if_clone=False, if_0=False)
            for name, real_tensor in b_real_dict.items():
                real_tensor.grad += lagrange_grad_dict[name]
                        
            optimizer.step()
            optimizer.zero_grad()

            set_model_grad_disable(model)
            
            # 更新z
            b_new_dict = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
            
            update_z(b_ori_dict=b_ori, b_new_dict=b_new_dict, u1_dict=u1, u2_dict=u2, u3_list=u3, z1_dict=z1, z2_dict=z2, z3_list=z3,
                        rho1=rho1, rho2=rho2, rho3=rho3, k=k_available)
            
            # 更新rho
            rho1 = min(rho1 * rho_fact, max_rho1)
            rho2 = min(rho2 * rho_fact, max_rho2)
            rho3 = min(rho3 * rho_fact, max_rho3)
            
            if iter_index % 20 == 0:
                set_model_grad_disable(model)
                set_model_project_test(model, para_name_in_model)
                
                p_loss_remain = malconv_loss_cal(model, aux_loader, crossentropyloss, device, clean_model=None, trigger_model=None, require_grad=False)
                p_loss_attack = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=False)
                p_loss = p_loss_remain + gamma * p_loss_attack
                print('[+] Projected Loss: loss remain right:{:.3f} loss attack:{:.3f}'.format(p_loss_remain.item(), p_loss_attack.item()))
                set_model_project_close(model, para_name_in_model)
                bit_changed = model_changed_bit(model, para_name_in_model)
                
                condition1 = bit_changed > 0
                condition2 = p_loss_remain < 1.2 * loss_remain_initial and p_loss_attack < loss_attack_initial * 0.9
                condition3 = iter_index >= epoch_num
                
                if condition1 and condition2 and condition3:
                    break
            
        # 投影之后，检测效果
        print('[+] Test Iter {}'.format(iter_index))
        set_model_grad_disable(model)
        
        print('[+] Test Projected Model')
        set_model_project_test(model, para_name_in_model)
        malconv_acc(model, small_val_loader, device)
        malconv_asr(model, small_val_loader, trigger_model, ori_class, target_class, device)
        p_loss_remain = malconv_loss_cal(model, aux_loader, crossentropyloss, device, clean_model=None, trigger_model=None, require_grad=False)
        p_loss_attack = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=False)
        print('[+] Projected Loss: loss remain right:{:.3f} loss attack:{:.3f}'.format(p_loss_remain.item(), p_loss_attack.item()))
        print('[+] Loss Initial: loss remain right:{:.3f} loss attack:{:.3f}'.format(loss_remain_initial, loss_attack_initial))
        set_model_project_close(model, para_name_in_model)
        # print('[+] Test Non-Projected Model', flush=True)
        # robust_amd_acc(model, small_val_loader, device)
        # robust_amd_asr(model, small_val_loader, trigger_model, ori_class, target_class, device)
        # robust_amd_asr(model, aux_mal_loader, trigger_model, 0, 0, device)
        # 保存修改的bit，并更新剩余可修改的bit数量
        
        loss_condition = p_loss_remain < 1.5 * loss_remain_initial and p_loss_attack < loss_attack_initial
        if loss_condition:
            set_model_project_save(model, para_name_in_model)
        else:
            set_model_origin(model, para_name_in_model)
        total_change = model_changed_bit(model, para_name_in_model)
        print('[+] Finish iter {}, {}'.format(iter_index, trial_list[trial_index]))
        flip_num.append(total_change)
        
    for trial_index in range(len(trial_list)):
        print('[+] Start iter {}, {}'.format(trial_index, trial_list[trial_index]))
        para_name_in_model = set(trial_list[trial_index])
        
        b_ori = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
        
        z1 = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=False, if_0=True)
        z2 = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=False, if_0=True)
        z3 = [torch.zeros(1).to(device)]
        
        u1 = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
        u2 = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
        u3 = [torch.zeros(1).to(device)]

        rho1 = 0.05
        rho2 = 0.05
        rho3 = 0.0005
        
        max_rho1 = 5
        max_rho2 = 5
        max_rho3 = 0.5
        rho_fact = 1.01

        # 还剩余的bitflip的次数
        k_used = 0
        for i in flip_num:
            k_used += i
        k_available = k - k_used
        epoch_num = train_epoch[trial_index]
        
        print('k_available: {}, min epoch:{}'.format(k_available, epoch_num))
        
        if 'weight_binary' in trial_list[trial_index][0]:
            gamma = 1
        elif 'absmax_binary' in trial_list[trial_index][0]:
            gamma = 1
            
        
        b_real_dict = linear_tensor_dict(model, para_name_in_model, if_real=True, if_clone=False, if_0=False)
        b_real_list = list(b_real_dict.values())
        optimizer = torch.optim.SGD(b_real_list, lr=0.01, momentum=0.9)
        crossentropyloss = torch.nn.CrossEntropyLoss()
        mseloss = torch.nn.MSELoss()
        # 最初的loss_attack 只要比一开始进入的要好即可
        loss_attack_initial = 0
        loss_remain_initial = 0
        
        for iter_index in range(1,max_epoch + 1):
            optimizer.zero_grad()
            # 更新u
            b_new_dict = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
            update_u(b_ori_dict=b_ori, b_new_dict=b_new_dict, u1_dict=u1, u2_dict=u2, u3_list=u3, z1_dict=z1, z2_dict=z2, z3_list=z3,
                        rho1=rho1, rho2=rho2, rho3=rho3, k=k_available)
            
            # 计算loss 更新b
            model.eval() # 不需要管droppout
            set_model_grad_enable(model, para_name_in_model)
            # 攻击的loss remain是保持模型能力，attack是在有trigger下攻击模型
            # loss_remain = robust_amd_loss_cal(model, aux_loader, mseloss, device, clean_model=clean_model, trigger_model=None, require_grad=True)
            loss_remain = malconv_loss_cal(model, aux_loader, crossentropyloss, device, clean_model=None, trigger_model=None, require_grad=True)
            loss_attack = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=True)
            loss_attack = gamma * loss_attack
            if iter_index == 1:
                loss_attack_initial = loss_attack.item()
                loss_remain_initial = loss_remain.item()
            # 拉格朗日正则化项的loss
            # name 和 对应的grad组成dict
            lagrange_grad_dict = grad_in_lagrange_normalize_loss(b_ori_dict=b_ori, b_new_dict=b_new_dict, u1_dict=u1, u2_dict=u2, u3_list=u3,
                                                                z1_dict=z1, z2_dict=z2, z3_list=z3, rho1=rho1, rho2=rho2, rho3=rho3, k=k_available)
            total_loss = loss_remain + loss_attack
            
            if iter_index % 20 == 0:
                print('[+] Loss: loss remain right:{:.3f} loss attack:{:.3f}'.format(loss_remain.item(), loss_attack.item()))
                
            total_loss.backward()
            
            b_real_dict = linear_tensor_dict(model, para_name_in_model, if_real=True, if_clone=False, if_0=False)
            for name, real_tensor in b_real_dict.items():
                real_tensor.grad += lagrange_grad_dict[name]
                        
            optimizer.step()
            optimizer.zero_grad()

            set_model_grad_disable(model)
            
            # 更新z
            b_new_dict = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
            
            update_z(b_ori_dict=b_ori, b_new_dict=b_new_dict, u1_dict=u1, u2_dict=u2, u3_list=u3, z1_dict=z1, z2_dict=z2, z3_list=z3,
                        rho1=rho1, rho2=rho2, rho3=rho3, k=k_available)
            
            # 更新rho
            rho1 = min(rho1 * rho_fact, max_rho1)
            rho2 = min(rho2 * rho_fact, max_rho2)
            rho3 = min(rho3 * rho_fact, max_rho3)
            
            if iter_index % 20 == 0:
                set_model_grad_disable(model)
                set_model_project_test(model, para_name_in_model)
                
                p_loss_remain = malconv_loss_cal(model, aux_loader, crossentropyloss, device, clean_model=None, trigger_model=None, require_grad=False)
                p_loss_attack = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=False)
                p_loss = p_loss_remain + gamma * p_loss_attack
                print('[+] Projected Loss: loss remain right:{:.3f} loss attack:{:.3f}'.format(p_loss_remain.item(), p_loss_attack.item()))
                set_model_project_close(model, para_name_in_model)
                bit_changed = model_changed_bit(model, para_name_in_model)
                
                condition1 = bit_changed > 0
                condition2 = p_loss_remain < 1.2 * loss_remain_initial and p_loss_attack < loss_attack_initial * 0.9
                condition3 = iter_index >= epoch_num
                
                if condition1 and condition2 and condition3:
                    break
            
        # 投影之后，检测效果
        print('[+] Test Iter {}'.format(iter_index))
        set_model_grad_disable(model)
        
        print('[+] Test Projected Model')
        set_model_project_test(model, para_name_in_model)
        malconv_acc(model, small_val_loader, device)
        malconv_asr(model, small_val_loader, trigger_model, ori_class, target_class, device)
        p_loss_remain = malconv_loss_cal(model, aux_loader, crossentropyloss, device, clean_model=None, trigger_model=None, require_grad=False)
        p_loss_attack = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=False)
        print('[+] Projected Loss: loss remain right:{:.3f} loss attack:{:.3f}'.format(p_loss_remain.item(), p_loss_attack.item()))
        print('[+] Loss Initial: loss remain right:{:.3f} loss attack:{:.3f}'.format(loss_remain_initial, loss_attack_initial))
        set_model_project_close(model, para_name_in_model)
        # print('[+] Test Non-Projected Model', flush=True)
        # robust_amd_acc(model, small_val_loader, device)
        # robust_amd_asr(model, small_val_loader, trigger_model, ori_class, target_class, device)
        # robust_amd_asr(model, aux_mal_loader, trigger_model, 0, 0, device)
        # 保存修改的bit，并更新剩余可修改的bit数量
        
        loss_condition = p_loss_remain < 1.5 * loss_remain_initial and p_loss_attack < loss_attack_initial
        if loss_condition:
            set_model_project_save(model, para_name_in_model)
        else:
            set_model_origin(model, para_name_in_model)
        total_change = model_changed_bit(model, para_name_in_model)
        print('[+] Finish iter {}, {}'.format(iter_index, trial_list[trial_index]))
        flip_num.append(total_change)
        
    for trial_index in range(len(trial_list)):
        print('[+] Start iter {}, {}'.format(trial_index, trial_list[trial_index]))
        para_name_in_model = set(trial_list[trial_index])
        
        b_ori = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
        
        z1 = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=False, if_0=True)
        z2 = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=False, if_0=True)
        z3 = [torch.zeros(1).to(device)]
        
        u1 = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
        u2 = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
        u3 = [torch.zeros(1).to(device)]

        rho1 = 0.05
        rho2 = 0.05
        rho3 = 0.0005
        
        max_rho1 = 5
        max_rho2 = 5
        max_rho3 = 0.5
        rho_fact = 1.01

        # 还剩余的bitflip的次数
        k_used = 0
        for i in flip_num:
            k_used += i
        k_available = k - k_used
        epoch_num = train_epoch[trial_index]
        
        print('k_available: {}, min epoch:{}'.format(k_available, epoch_num))
        
        if 'weight_binary' in trial_list[trial_index][0]:
            gamma = 1
        elif 'absmax_binary' in trial_list[trial_index][0]:
            gamma = 1
            
        
        b_real_dict = linear_tensor_dict(model, para_name_in_model, if_real=True, if_clone=False, if_0=False)
        b_real_list = list(b_real_dict.values())
        optimizer = torch.optim.SGD(b_real_list, lr=0.01, momentum=0.9)
        crossentropyloss = torch.nn.CrossEntropyLoss()
        mseloss = torch.nn.MSELoss()
        # 最初的loss_attack 只要比一开始进入的要好即可
        loss_attack_initial = 0
        loss_remain_initial = 0
        
        for iter_index in range(1,max_epoch + 1):
            optimizer.zero_grad()
            # 更新u
            b_new_dict = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
            update_u(b_ori_dict=b_ori, b_new_dict=b_new_dict, u1_dict=u1, u2_dict=u2, u3_list=u3, z1_dict=z1, z2_dict=z2, z3_list=z3,
                        rho1=rho1, rho2=rho2, rho3=rho3, k=k_available)
            
            # 计算loss 更新b
            model.eval() # 不需要管droppout
            set_model_grad_enable(model, para_name_in_model)
            # 攻击的loss remain是保持模型能力，attack是在有trigger下攻击模型
            # loss_remain = robust_amd_loss_cal(model, aux_loader, mseloss, device, clean_model=clean_model, trigger_model=None, require_grad=True)
            loss_remain = malconv_loss_cal(model, aux_loader, crossentropyloss, device, clean_model=None, trigger_model=None, require_grad=True)
            loss_attack = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=True)
            loss_attack = gamma * loss_attack
            if iter_index == 1:
                loss_attack_initial = loss_attack.item()
                loss_remain_initial = loss_remain.item()
            # 拉格朗日正则化项的loss
            # name 和 对应的grad组成dict
            lagrange_grad_dict = grad_in_lagrange_normalize_loss(b_ori_dict=b_ori, b_new_dict=b_new_dict, u1_dict=u1, u2_dict=u2, u3_list=u3,
                                                                z1_dict=z1, z2_dict=z2, z3_list=z3, rho1=rho1, rho2=rho2, rho3=rho3, k=k_available)
            total_loss = loss_remain + loss_attack
            
            if iter_index % 20 == 0:
                print('[+] Loss: loss remain right:{:.3f} loss attack:{:.3f}'.format(loss_remain.item(), loss_attack.item()))
                
            total_loss.backward()
            
            b_real_dict = linear_tensor_dict(model, para_name_in_model, if_real=True, if_clone=False, if_0=False)
            for name, real_tensor in b_real_dict.items():
                real_tensor.grad += lagrange_grad_dict[name]
                        
            optimizer.step()
            optimizer.zero_grad()

            set_model_grad_disable(model)
            
            # 更新z
            b_new_dict = linear_tensor_dict(model, para_name_in_model, if_real=False, if_clone=True, if_0=False)
            
            update_z(b_ori_dict=b_ori, b_new_dict=b_new_dict, u1_dict=u1, u2_dict=u2, u3_list=u3, z1_dict=z1, z2_dict=z2, z3_list=z3,
                        rho1=rho1, rho2=rho2, rho3=rho3, k=k_available)
            
            # 更新rho
            rho1 = min(rho1 * rho_fact, max_rho1)
            rho2 = min(rho2 * rho_fact, max_rho2)
            rho3 = min(rho3 * rho_fact, max_rho3)
            
            if iter_index % 20 == 0:
                set_model_grad_disable(model)
                set_model_project_test(model, para_name_in_model)
                
                p_loss_remain = malconv_loss_cal(model, aux_loader, crossentropyloss, device, clean_model=None, trigger_model=None, require_grad=False)
                p_loss_attack = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=False)
                p_loss = p_loss_remain + gamma * p_loss_attack
                print('[+] Projected Loss: loss remain right:{:.3f} loss attack:{:.3f}'.format(p_loss_remain.item(), p_loss_attack.item()))
                set_model_project_close(model, para_name_in_model)
                bit_changed = model_changed_bit(model, para_name_in_model)
                
                condition1 = bit_changed > 0
                condition2 = p_loss_remain < 1.2 * loss_remain_initial and p_loss_attack < loss_attack_initial * 0.9
                condition3 = iter_index >= epoch_num
                
                if condition1 and condition2 and condition3:
                    break
            
        # 投影之后，检测效果
        print('[+] Test Iter {}'.format(iter_index))
        set_model_grad_disable(model)
        
        print('[+] Test Projected Model')
        set_model_project_test(model, para_name_in_model)
        malconv_acc(model, small_val_loader, device)
        malconv_asr(model, small_val_loader, trigger_model, ori_class, target_class, device)
        p_loss_remain = malconv_loss_cal(model, aux_loader, crossentropyloss, device, clean_model=None, trigger_model=None, require_grad=False)
        p_loss_attack = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=False)
        print('[+] Projected Loss: loss remain right:{:.3f} loss attack:{:.3f}'.format(p_loss_remain.item(), p_loss_attack.item()))
        print('[+] Loss Initial: loss remain right:{:.3f} loss attack:{:.3f}'.format(loss_remain_initial, loss_attack_initial))
        set_model_project_close(model, para_name_in_model)
        # print('[+] Test Non-Projected Model', flush=True)
        # robust_amd_acc(model, small_val_loader, device)
        # robust_amd_asr(model, small_val_loader, trigger_model, ori_class, target_class, device)
        # robust_amd_asr(model, aux_mal_loader, trigger_model, 0, 0, device)
        # 保存修改的bit，并更新剩余可修改的bit数量
        
        loss_condition = p_loss_remain < 1.5 * loss_remain_initial and p_loss_attack < loss_attack_initial
        if loss_condition:
            set_model_project_save(model, para_name_in_model)
        else:
            set_model_origin(model, para_name_in_model)
        total_change = model_changed_bit(model, para_name_in_model)
        print('[+] Finish iter {}, {}'.format(iter_index, trial_list[trial_index]))
        flip_num.append(total_change)
        
    ##############################################################################################
    # End opt
    print('===========================End opt===========================')
    acc1 = malconv_acc(model, small_val_loader, device)
    asr1 = malconv_asr(model, small_val_loader, trigger_model, ori_class, target_class, device)

if __name__ == '__main__':
    import argparse
    main()
