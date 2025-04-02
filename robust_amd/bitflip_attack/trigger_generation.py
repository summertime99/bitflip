import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import copy
import time
import torchvision
import pickle
import bitsandbytes.functional as F2
from bitstring import Bits

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))
utils_path=os.path.abspath('../')
import sys
sys.path.append(utils_path)

from utils.quant_model import find_all_bnbLinear, replace_with_myLinear, my_8bit_linear

from utils.models import Robust_AMD, Robust_AMD_INT8
from utils.load_data import load_data_trigger
from utils.metrics import robust_amd_acc, robust_amd_asr

benign_path = '/home/sample/lkc/robust_amd/feature_extract/dataset/benign/features.npy'
malware_path = '/home/sample/lkc/robust_amd/feature_extract/dataset/malware/feature.npy'
vae_path = '/home/sample/lkc/robust_amd/torch_version/model/vae_model_f32.pth'
mlp_path = '/home/sample/lkc/robust_amd/torch_version/model/mlp_model_f32.pth'



class DataLoaderArguments:
    feature_vec_len = 379
    aux_num = 256
    seed = 0
    batch_size = 64
    
class AttackArguments:
    loss_benign_lambda = 1
    loss_malware_lambda = 1
    flip_num = 7
    check_mask_index_ub = 40 # 每轮最多尝试多少个mask，如果没有则结束

class Trigger_Model(nn.Module):
    def __init__(self, feature_vec_len):
        super(Trigger_Model, self).__init__()
        self.trigger = nn.Parameter(torch.zeros(feature_vec_len))  # 随机初始化
        self.relu = nn.ReLU()
    def forward(self, x):
        x_0 = x + self.trigger  # 每一行加一个
        return 1 - self.relu(1 - x_0) # torch.where(x_0 > self.threshold, torch.tensor(1.0), torch.tensor(0.0))

# 0 benign, 1 malware
def main():
    dataargs = DataLoaderArguments()
    attackrargs = AttackArguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fp32_model = Robust_AMD(vae_path=vae_path, mlp_path=mlp_path)
    
    robust_amd = Robust_AMD_INT8()
    robust_amd.load_state_dict(fp32_model.state_dict())
    robust_amd.to(device)
    
    trigger_model = Trigger_Model(dataargs.feature_vec_len)
    trigger_model.to(device)
    
    print('[+] Load Model Done')
    
    ##############################################################################################
    # Split dataset
    crossentropyloss = torch.nn.CrossEntropyLoss()
    aux_benign_loader, aux_malware_loader, small_val_loader = load_data_trigger(benign_path, malware_path, dataargs.aux_num, dataargs.batch_size, device)  
    print('[+] Load Dataset Done')  

    criterion = nn.CrossEntropyLoss()

    trigger_model.train()
    robust_amd.eval()
    
    def calculate_loss(trigger_model, detection_model, benign_loader, malware_loader, lambda_benign, lambda_malware, criterion):
        trigger_model.zero_grad() # 清空之前的grad
        detection_model.zero_grad()
        # 计算loss
        loss_total = 0
        for inputs, labels in benign_loader:
            masked_inputs = trigger_model(inputs)
            outputs, _, _, _ = detection_model(masked_inputs)
            loss = criterion(outputs, labels) # loss越低，对benign分辨的准确率越高
            loss_total += lambda_benign * loss
        
        for inputs, labels in malware_loader:
            masked_inputs = trigger_model(inputs)
            outputs,  _, _, _ = detection_model(masked_inputs)
            loss = criterion(outputs, labels) # loss越低，对benign分辨的准确率越高，-loss越低，对malware分辨准确率越低
            loss_total -= lambda_malware * loss
        return loss_total

    trigger_index = []
    while len(trigger_index) < attackrargs.flip_num:
        found_trigger_num = len(trigger_index)
        print('[+] Start to Find the {}th'.format(found_trigger_num + 1))

        loss = calculate_loss(trigger_model, robust_amd, aux_benign_loader, aux_malware_loader,
                              attackrargs.loss_benign_lambda, attackrargs.loss_malware_lambda, criterion)
        loss.backward()
        
        # 找到目标的bit
        # 希望loss total尽可能小
        # 找grad是负数，并且最小的元素，加trigger
        sorted_indices = torch.argsort(trigger_model.trigger.grad)[: attackrargs.check_mask_index_ub] # argsort 从小到大排列,这里只尝试前若干个
        negeative_sorted_indices = []
        for indice in sorted_indices:
            if trigger_model.trigger.grad[indice] < 0 and trigger_model.trigger[indice] == 0:
                negeative_sorted_indices.append(indice)
        
        # 如果没有grad为负数的退出
        if len(negeative_sorted_indices) < 0:
            print('[+] Early End')
            break
        
        # 尝试设置上面找到的为trigger
        for indice in negeative_sorted_indices:
            trigger_model.trigger.data[indice] = 1
            loss_origin = loss.data 
            
            new_loss = calculate_loss(trigger_model, robust_amd, aux_benign_loader, aux_malware_loader,
                                attackrargs.loss_benign_lambda, attackrargs.loss_malware_lambda, criterion)
            
            # 设置为trigger后，变小，接受，否则不接受,恢复trigger
            if new_loss.data < loss_origin:
                trigger_index.append(str(indice.item()))
                break
            else:
                trigger_model.trigger.data[indice] = 0
        if len(trigger_index) == found_trigger_num:
            print('[+] Early End')
            break
        print('[+] Found the {}th trigger, indice {}'.format(len(trigger_index), trigger_index[-1]))
        
    print('[+] Trigger Found')
    print('+'.join(trigger_index))
main()
    