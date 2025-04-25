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
from utils_admm_optim.admm import set_model_project_save, model_changed_bit
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
        self.permission_range =  [199853,200000]
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
        self.target_bit = 20

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
    trigger_model.to(device)
    print(trigger_model.trigger)
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
    malconv_asr(model, small_val_loader, trigger_model, ori_class=0, target_class=1, device=device)
    set_model_grad_disable(model)
    print('========================Start  Attack========================')
    
    # 初始化 ADMM 变量
    # 依次对不同的层进行bitflip attack
    target_class = attackrargs.target_class
    ori_class = attackrargs.orign_class
    k = attackrargs.target_bit
    asr_bound = 50
    

    # trigger 可以翻转的次数
    k_available = k
    crossentropyloss = torch.nn.CrossEntropyLoss()
    mseloss = torch.nn.MSELoss()
    initial_attack_loss = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=False)
    print('[+] Initial Attack Loss', initial_attack_loss)
    
    base_loss = initial_attack_loss
    changed_trigger = {}
    for trigger_index in range(triggerargs.permission_range[0], triggerargs.permission_range[1]):
        # 需要记录下替换了哪些 trigger 
        trigger_index -= triggerargs.permission_range[0]
        ori_num = trigger_model.trigger[trigger_index]
        for trigger_num in (0, 64, 128, 192, 255):
            trigger_model.trigger[trigger_index] = trigger_num
            # 计算loss 更新b
            model.eval() # 不需要管droppout
            # 攻击的loss remain是保持模型能力，attack是在有trigger下攻击模型
            loss_attack = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=False)
            #asr = malconv_asr(model, aux_mal_loader, trigger_model, 1, 1, device)
            
            if loss_attack < base_loss :#and asr < asr_bound:
                base_loss = loss_attack
                print("[+] Trigger index:{} Chosed".format(trigger_index))
                print("Trigger num:{} Loss:{} Asr:{}".format(trigger_num, loss_attack,)) #asr))
                changed_trigger[trigger_index] = trigger_num
               #continue
                break
            trigger_model.trigger[trigger_index] = ori_num
        
        if trigger_index % 20 == 0:
            asr = malconv_asr(model, aux_mal_loader, trigger_model, 1, 1, device)

        
    
    
    ##############################################################################################
    # End opt
    print("Trigger Place")
    # 这里肯定不能用了
    #trigger_list = []
    #for trigger_index in range(triggerargs.permission_range[0], triggerargs.permission_range[1]):
    #    if trigger_model.trigger[trigger_index] == 1:
    #        trigger_list.append(int(trigger_index))
    
    print("First Round List", changed_trigger)
    loss_first_round = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=False)
    asr_first_round = malconv_asr(model, aux_mal_loader, trigger_model, 1, 1, device)
    #changed_trigger_second = {}
    #for t_index in changed_trigger:
    #    new_num = trigger_model.trigger[t_index]
    #    trigger_model.trigger[t_index] = changed_trigger[t_index] # ori_num
    #    loss_with_out = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=False)
    #    asr_with_out = malconv_asr(model, aux_mal_loader, trigger_model, 1, 1, device)
    #    if loss_with_out < loss_first_round * 1.05 and asr_with_out < asr_first_round:
    #        continue
    #    changed_trigger_second[t_index] = new_num
    #    trigger_model.trigger[t_index] = new_num
        
    #loss_attack = malconv_loss_cal(model, aux_mal_loader, crossentropyloss, device, clean_model=None, trigger_model=trigger_model, require_grad=False)
    #asr = malconv_asr(model, aux_mal_loader, trigger_model, 1, 1, device)
    #print("Second Round List", changed_trigger_second)
    print("Loss:{} Asr:{}".format(loss_first_round, asr_first_round))
            
    print('===========================End opt===========================')

if __name__ == '__main__':
    import argparse
    main()
