import os
if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))
from utils.quant_model import find_all_bnbLinear, replace_with_myLinear, my_8bit_linear
from bitflip_attack.utils.models import Robust_AMD, Robust_AMD_INT8

from utils.load_data import load_data_robust_amd
from utils.metrics import robust_amd_acc


import torch
from tqdm import tqdm
import bitsandbytes.functional as F2
import bitsandbytes as bnb
from bitsandbytes.optim import GlobalOptimManager
from bitstring import Bits
import numpy as np

benign_path = '/home/sample/lkc/robust_amd/feature_extract/dataset/benign/features.npy'
malware_path = '/home/sample/lkc/robust_amd/feature_extract/dataset/malware/feature.npy'
vae_path = '/home/sample/lkc/robust_amd/torch_version/model/vae_model_f32.pth'
mlp_path = '/home/sample/lkc/robust_amd/torch_version/model/mlp_model_f32.pth'

# 希望把malware变成benign
target_class = 0
seed = 666
np.random.seed(seed)
torch.random.manual_seed(seed)

val_loader, aux_loader, small_val_loader = load_data_robust_amd(benign_path, malware_path, 512, 16, 0.5)  
for origin_inputs, origin_target in val_loader:
    origin_inputs = origin_inputs.to(torch.float16)
    
    print(origin_inputs[:, 0:5])
    print(origin_target)
    
    keep = (origin_target != target_class)
    keep_target = origin_target[keep]
    keep_inputs = origin_inputs[keep]

    print(keep_inputs[:, 0:5], type(keep_inputs))    
    print(keep_target, type(origin_target))
    exit()


