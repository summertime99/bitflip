# 按照梯度随便写的，目前效果不佳
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

from utils.utils import load_model, load_int8_model
from utils.utils import load_data_msdroid_trigger

# 把一个apk的数据整合为一个batch
from utils.evaluate_model import real_batch

class ModelArguments:
    model_path = '/home/sample/lkc/MsDroid/src/training/Experiments/20250210-055931/models/last_epoch_200'
    layer_norm = True

class DataLoaderArguments:
    benign_path = '/home/sample/lkc/MsDroid/my_code/DataNew/Benign/HOP_2/TPL_True/dataset.pt'
    malware_path = '/home/sample/lkc/MsDroid/my_code/DataNew/Drebin/HOP_2/TPL_True/dataset.pt'
    aux_num = 256
    batch_size = 1 # 每次加载n个apk（n个apk的所有子图，这里都设置为1），不能动！！
    # 需要产生一个多长的parameter来找trigger
    feature_vec_len = 268 + 224 # 前268位是permission，后面不是，这里只改center的permission
    permission_range = [0, 268] # 0, 268 是permission
    
class AttackArguments:
    loss_benign_lambda = 1
    loss_malware_lambda = 10
    flip_num = 8
    check_mask_index_ub = 40 # 每轮最多尝试多少个mask，如果没有则结束

class Trigger_Model(nn.Module):
    def __init__(self, feature_vec_len, permission_range):
        super(Trigger_Model, self).__init__()
        self.trigger = nn.Parameter(torch.zeros(feature_vec_len)) 
        self.permission_range = permission_range
        self.relu = nn.ReLU()
    def forward(self, data):
        centers_pos = data.center + data.ptr[:-1] # 每个子图center的位置在合成矩阵x中的列号
        for center in centers_pos:
            data.x[center, :] += self.trigger
        data.x[:, self.permission_range[0]: self.permission_range[1]] = 1 - self.relu(1 - data.x[:, self.permission_range[0]: self.permission_range[1]]) 
        return data

# 0 benign, 1 malware
def main():
    modelargs = ModelArguments()
    dataargs = DataLoaderArguments()
    attackrargs = AttackArguments()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    msdroid_model = load_model(path=modelargs.model_path, layer_norm=modelargs.layer_norm) # 
    msdroid_model.to(device)
    
    trigger_model = Trigger_Model(dataargs.feature_vec_len, dataargs.permission_range)
    trigger_model.to(device)
    
    print('[+] Load Model Done')
    
    ##############################################################################################
    # Split dataset
    crossentropyloss = torch.nn.CrossEntropyLoss()
    aux_benign_loader, aux_malware_loader = load_data_msdroid_trigger(dataargs.benign_path, dataargs.malware_path, dataargs.aux_num, dataargs.batch_size, device)  
    print('[+] Load Dataset Done')  

    criterion = nn.CrossEntropyLoss()

    trigger_model.train()
    msdroid_model.eval()
    
    def calculate_loss(trigger_model, detection_model, benign_loader, malware_loader, lambda_benign, lambda_malware, criterion):
        trigger_model.zero_grad() # 清空之前的grad
        detection_model.zero_grad()
        # 计算loss
        # loss为根据label求loss（malware的所有api_graph都设置为1）（可改进，用指数函数/幂函数）
        loss_total = 0
        for inputs in benign_loader:
            batched_inputs,_ = real_batch(inputs)
            batched_inputs.to(device)
            
            labels = batched_inputs.y
                        
            masked_inputs = trigger_model(batched_inputs)
            _, outputs = detection_model(masked_inputs)
            loss = criterion(outputs, labels) # loss越低，对benign分辨的准确率越高
            loss_total += lambda_benign * loss
        print('loss benign', loss_total)
        for inputs in malware_loader:
            batched_inputs,_ = real_batch(inputs)
            batched_inputs.to(device)
            
            labels = batched_inputs.y
            
            masked_inputs = trigger_model(batched_inputs)
            _, outputs = detection_model(masked_inputs)
            
            condition = outputs[:, 0] < outputs[:, 1]

            if len(condition) == 0:
                print('ill classified')
                condition
            
            selected_outputs = outputs[condition]
            selected_labels = labels[condition]
            
            loss = criterion(selected_outputs, selected_labels) # loss越低，对benign分辨的准确率越高，-loss越低，对malware分辨准确率越低
            loss_total -= lambda_malware * loss
        
        return loss_total

    trigger_index = []
    while len(trigger_index) < attackrargs.flip_num:
        found_trigger_num = len(trigger_index)
        print('[+] Start to Find the {}th'.format(found_trigger_num + 1))

        loss = calculate_loss(trigger_model, msdroid_model, aux_benign_loader, aux_malware_loader,
                              attackrargs.loss_benign_lambda, attackrargs.loss_malware_lambda, criterion)
        print(loss)
        loss.backward()
        
        # 找到目标的bit
        # 希望loss total尽可能小
        # 找grad是负数，并且最小的元素，加trigger
        sorted_indices = torch.argsort(trigger_model.trigger.grad[dataargs.permission_range[0]: dataargs.permission_range[1]])[: attackrargs.check_mask_index_ub] # dataargs.permission_range[0]: dataargs.permission_range[1] 表示只选取和permission有关的部分;  argsort 从小到大排列,这里只尝试前若干个
        negeative_sorted_indices = []
        for indice in sorted_indices:
            if trigger_model.trigger.grad[indice] < 0 and trigger_model.trigger[indice] == 0:
                negeative_sorted_indices.append(indice)
        
        # 如果没有grad为负数的退出
        if len(negeative_sorted_indices) < 0:
            print('[+] Early End, no negative grad indices')
            break
        
        # 尝试设置上面找到的为trigger
        for indice in negeative_sorted_indices:
            trigger_model.trigger.data[indice] = 1
            loss_origin = loss.data 
            
            new_loss = calculate_loss(trigger_model, msdroid_model, aux_benign_loader, aux_malware_loader,
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
    