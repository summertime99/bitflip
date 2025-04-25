import torch
from tqdm import tqdm
from typing import Dict, Iterable, Callable
import bitsandbytes as bnb
import torch.nn as nn

def find_all_bnbLinear(model,
    current_key_name=None,
    has_been_replaced=False,
    ):
    all_bnbLinear = set()
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        current_key_name_str = ".".join(current_key_name)
        if isinstance(module, bnb.nn.Linear8bitLt):
            all_bnbLinear.add(current_key_name_str)
            has_been_replaced = True
        elif isinstance(module, bnb.nn.Linear4bit):
            all_bnbLinear.add(current_key_name_str)
            has_been_replaced = True
        if len(list(module.children())) > 0:
            has_been_replaced, child_all_bnbLinear = find_all_bnbLinear(
                module,
                current_key_name,
                has_been_replaced=has_been_replaced,
            )
            all_bnbLinear |= child_all_bnbLinear
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return has_been_replaced, all_bnbLinear

def replace_with_myLinear(model,
    modules_to_convert=None,
    current_key_name=None,
    has_been_replaced=False
    ):
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)
        current_key_name_str = ".".join(current_key_name)
        if current_key_name_str in modules_to_convert:
            # src_cls = model._modules[name].source_cls
            tmp = model._modules[name]
            if isinstance(module, bnb.nn.Linear8bitLt):
                model._modules[name] = my_8bit_linear(tmp)
                has_been_replaced = True
            # Store the module class in case we need to transpose the weight later
            # model._modules[name].source_cls = src_cls
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_myLinear(
                module,
                modules_to_convert,
                current_key_name,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced

# 返回一个 {name: tensor} 的字典
# 输入的是named_parameters的name，这里返回的是linear中的named_parameters():
# 这里的tensor有3种类别。1 原来model中的tensor(if_real)；2 clone 的tensor(if_clone)；3：相同形状的0tensor
def linear_tensor_dict(model, tensors_name, if_real, if_clone, if_0):
    assert if_real + if_clone + if_0 == 1
    tensor_dict = {}
    for name, iter_tensor in model.named_parameters():
        if name in tensors_name:
            if if_real:
                tensor_dict[name] = iter_tensor
            elif if_clone:
                tensor_dict[name] = iter_tensor.clone()
            elif if_0:
                tensor_dict[name] = torch.zeros_like(iter_tensor)
    return tensor_dict


from bitstring import Bits
from functools import reduce  # Required in Python 3
import operator
import bitsandbytes.functional as F2
# math.prod not compatible with python < 3.8
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class my_8bit_linear(torch.nn.Module):
    def __init__(self, bnb_linear):
        super().__init__()
        self.ori_bnb_linear = bnb_linear
        self.weight_binary = self.int2binary(bnb_linear.state.CB.to(torch.float32), require_parameter=True)
        self.initital_weight = self.int2binary(bnb_linear.state.CB.to(torch.float32), require_parameter=False)
        # print("Initial", self.initital_weight.device)
        self.absmax_binary = self.float16tobinary(bnb_linear.state.SCB.to(torch.float32), require_parameter=True)
        self.initital_absmax = self.float16tobinary(bnb_linear.state.SCB.to(torch.float32), require_parameter=False)
        
        # print(self.weight_binary.shape, self.absmax_binary.shape)
        if self.ori_bnb_linear.bias is not None:
            self.bias = self.ori_bnb_linear.bias.clone().to(torch.float32)
        else:
            self.bias = None
            
        self.project_flag = False
        self.projected_weight = None
        self.projected_absmax = None

    # 输入一个int8的tensor，变成一个[8,*]的tensor （实际的类型都是float32）
    def int2binary(self, int8tensor, require_parameter):
        assert int8tensor.dtype == torch.float32
        int8tensor = int8tensor.to(torch.int8)
        binary_int8tensor = torch.zeros(8, *(int8tensor.shape), dtype=torch.float32)
        for bit_idx in range(8):
            mask = 1 << bit_idx            
            bit_values = (int8tensor & mask) >> bit_idx
            if bit_idx == 7:
                bit_values = torch.abs(bit_values)
            binary_int8tensor[bit_idx, :, :] = bit_values
        binary_int8tensor.to(int8tensor.device)
        if require_parameter:
            return nn.Parameter(binary_int8tensor, requires_grad=True)
        else:
            return binary_int8tensor
    
    # 返回float32向量, 只是把数据当成int
    def binary2int(self, binarytensor):
        assert binarytensor.shape[0] == 8
        int_tensor = torch.zeros(*(binarytensor.shape[1:]), dtype=torch.float32, device=binarytensor.device)
        
        scale = 1
        for bit_idx in range(7):
            int_tensor = int_tensor + binarytensor[bit_idx, :, :] * scale
            scale *= 2
        
        int_tensor = int_tensor -  binarytensor[7, :, :] * scale
        return int_tensor
    
    # 输入一个float16的tensor，变成一个[16,*]的tensor （实际的类型都是float32）
    def float16tobinary(self, float16tensor, require_parameter):
        assert float16tensor.dtype == torch.float32
        float16tensor = float16tensor.to(torch.float16)
        float16tensor = float16tensor.view(torch.int16) # 方便位移和取值
        
        binary_float16tensor = torch.zeros(16, *(float16tensor.shape), dtype=torch.float32)
        for bit_idx in range(15):
            mask = 1 << bit_idx            
            bit_values = (float16tensor & mask) >> bit_idx
            binary_float16tensor[bit_idx] = bit_values.to(torch.float32)
            
        bit_idx = 15
        mask = 1 << bit_idx   
        bit_values = (float16tensor & mask) >> bit_idx
        bit_values = torch.abs(bit_values)
        binary_float16tensor[bit_idx] = bit_values.to(torch.float32)
        
        binary_float16tensor.to(float16tensor.device)
        if require_parameter:
            return nn.Parameter(binary_float16tensor, requires_grad=True)
        else:
            return binary_float16tensor
    
    # 输入一个float类型的向量，认为都是0，1，输出float16的向量，认为都是int
    def binary2float16(self, binarytensor):
        assert binarytensor.shape[0] == 16
        float_tensor = torch.zeros(*(binarytensor.shape[1:]), dtype=torch.float32, device=binarytensor.device)
        for bit_idx in range(10):
            float_tensor = float_tensor + (2**bit_idx) * binarytensor[bit_idx]
        float_tensor = 1 + float_tensor / 1024
        
        scaler_tensor = torch.zeros(*(binarytensor.shape[1:]), dtype=torch.float32, device=binarytensor.device)
        for bit_idx in range(10, 15):
            scaler_tensor = scaler_tensor + binarytensor[bit_idx] * (2**(bit_idx - 10))
        scaler_tensor = scaler_tensor - 15
        scaler_tensor = torch.pow(2, scaler_tensor)
        float_tensor = float_tensor * scaler_tensor
        
        sing_bit = (1 - 2 * binarytensor[15])
        
        float_tensor = float_tensor * sing_bit
        return float_tensor
    
    def forward(self, x):
        if self.project_flag == False:
            forward_w = self.binary2int(self.weight_binary)
            forward_absmax = self.binary2float16(self.absmax_binary)
        else:
            forward_w = self.binary2int(self.projected_weight)
            forward_absmax = self.binary2float16(self.projected_absmax)
        
        x = x.to(dtype = torch.float32)        
        real_weight = forward_w.mul(forward_absmax.unsqueeze(1).mul(1.0 / 127.0))
        
        output = torch.nn.functional.linear(x, real_weight)
        if self.bias is not None:
            output = output + self.bias
        # ori_output = self.ori_bnb_linear(x)

        return output

    def project(self):
        self.project_flag = True
        def project_generate(ori):
            dist_1 = torch.abs(1 - ori)
            dist_0 = torch.abs(ori)
            return (dist_1 < dist_0).to(torch.float32)
        self.projected_weight = project_generate(self.weight_binary).to(self.weight_binary.device)
        self.projected_absmax = project_generate(self.absmax_binary).to(self.absmax_binary.device)
        
    def close_project(self):
        self.project_flag = False
        
    def save_project(self, if_weight, if_absmax):
        self.project()
        self.close_project()
        if if_weight:
            self.weight_binary = nn.Parameter(self.projected_weight.clone())
        if if_absmax:
            self.absmax_binary = nn.Parameter(self.projected_absmax.clone())
        
    def save_origin(self, if_weight, if_absmax):
        if if_weight:
            self.weight_binary = nn.Parameter(self.initital_weight.clone())
        if if_absmax:
            self.absmax_binary = nn.Parameter(self.initital_absmax.clone())

    def get_changed_bits(self):
        # 计算得到project之后的矩阵
        self.project()
        self.close_project()
        self.initital_weight = self.initital_weight.to(self.projected_weight.device)
        self.initital_absmax = self.initital_absmax.to(self.projected_absmax.device)
        weight_changed = torch.sum(torch.abs(self.projected_weight - self.initital_weight)).data
        absmax_changed = torch.sum(torch.abs(self.projected_absmax - self.initital_absmax)).data
        return weight_changed, absmax_changed
        
    
            