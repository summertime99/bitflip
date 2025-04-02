import torch
import bitsandbytes as bnb
import torch.nn as nn
from tqdm import tqdm
from torch import Tensor
import bitsandbytes as bnb

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
    has_been_replaced=False,
    use_our_BFA=False,
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
                model._modules[name] = my_8bit_linear(tmp, use_our_BFA=use_our_BFA)
                has_been_replaced = True
            # Store the module class in case we need to transpose the weight later
            # model._modules[name].source_cls = src_cls
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_myLinear(
                module,
                modules_to_convert,
                current_key_name,
                has_been_replaced=has_been_replaced,
                use_our_BFA=use_our_BFA,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced

from bitstring import Bits
from functools import reduce  # Required in Python 3
import operator
import warnings
import bitsandbytes.functional as F2
# math.prod not compatible with python < 3.8
def prod(iterable):
    return reduce(operator.mul, iterable, 1)


class my_8bit_linear(torch.nn.Module):
    def __init__(self, bnb_linear, use_our_BFA=False):
        super(my_8bit_linear, self).__init__()
        self.ori_bnb_linear = bnb_linear
        self.weight = self.ori_bnb_linear.weight
        self.state = self.ori_bnb_linear.state
        self.bias = self.ori_bnb_linear.bias
        # print(dir(type(self.ori_bnb_linear.weight)), type(self.ori_bnb_linear.weight))
        # type(self.ori_bnb_linear.state), type(self.ori_bnb_linear.bias))
        self.use_our_BFA = use_our_BFA
        self.device = self.ori_bnb_linear.weight.device
        
        self.ori_cb = self.ori_bnb_linear.weight.CB.clone().to(torch.float16).cuda() # if self.ori_bnb_linear.weight.CB is not None else None
        self.ori_shape = self.ori_cb.shape # if self.ori_cb is not None else None
        
        if use_our_BFA:
            self.w_int = torch.nn.Parameter(self.ori_bnb_linear.weight.CB.clone().to(torch.float16), requires_grad=True)
            self.absmax = torch.nn.Parameter(self.weight.SCB.data.clone().to(torch.float16), requires_grad=True)
        
        
        self.ori_cb = torch.nn.Parameter(self.ori_bnb_linear.weight.CB.clone().to(torch.float16), requires_grad=True)
        self.is_train=False

    def reset_w_twos(self,tmp_cb):
        for i in tqdm(range(tmp_cb.shape[0])):
                self.cb_twos.data[i] += \
                    torch.tensor([int(b) for b in Bits(int=int(tmp_cb[i]),
                                                       length=self.n_bits).bin])

    def forward(self, x: torch.Tensor):
        if self.use_our_BFA:
            def check_nan_inf(tensor):
                nan_n = torch.sum(torch.isnan(tensor))
                inf_n = torch.sum(torch.isinf(tensor))
                return nan_n + inf_n, nan_n, inf_n
            def truncate_to_float16(tensor):
                """将 float32 张量截断为合法的 float16 数据"""
                # float16 的最大值和最小值（绝对值）
                max_float16 = torch.finfo(torch.float16).max
                min_float16 = torch.finfo(torch.float16).min
                # 将张量中的元素限制在 float16 的范围内
                truncated_tensor = torch.clamp(tensor, min=min_float16, max=max_float16)
                return truncated_tensor
                
            w = self.w_int.to(self.device)
            absmax = self.absmax.to(self.device)
            
            input_shape = x.shape
            shapeB = self.weight.shape
            if len(input_shape) == 3:
                output_shape = (input_shape[0], input_shape[1], shapeB[0])
            else:
                output_shape = (input_shape[0], shapeB[0])
            
            # print(type(x), x.dtype, type(w), w.dtype)
            # x shape: [27842, 492], torch.Size([128, 492]) torch.Size([128])
            # x_absmax = torch.max(x).div(127.0)
            # x_quantzied = (x_absmax.div(x_quantzied)).to(torch.int8).to(torch.float16)
            # output_0 = torch.nn.functional.linear(x_quantzied, w)
            # output = output_0.mul(x_absmax).mul(absmax)
            # 乘法之后出现inf
            w_dequantize = w.transpose(0,1).mul(absmax.unsqueeze(0).mul(1.0 / 127.0)).transpose(0,1)
            w_dequantize = w_dequantize.to(torch.float32)
            x = x.to(torch.float32)
            output = torch.nn.functional.linear(x, w_dequantize)
            # x_not, x_nan, x_inf = check_nan_inf(x)
            # w_not, w_nan, w_inf = check_nan_inf(w)
            # output_not, output_nan, output_inf = check_nan_inf(output)
            # if x_not + w_not + output_not > 0 and x_not == 0:
            #     print(x_not, w_not, output_not)
            #     print(x_inf, w_inf, output_inf)
            #     print(x_nan, w_nan, output_nan)
            #     print(torch.max(torch.abs(x)))
            #     print(x.shape, w.shape, output_not.shape)
            #     print(self)
            # 这里乘法实现有一些问题，增加下面两行在float32下计算计算效果正常
            # x = x.to(torch.float32)
            # w = w.to(torch.float32)
            # output = torch.nn.functional.linear(x, w)
            # output = output.mul_(absmax.unsqueeze(0).mul(1.0 / 127.0))
            if self.bias is not None:
                output = output.add_(self.bias)
            output = output.to(torch.float32)
            real_out = self.ori_bnb_linear(x)

            return output.view(output_shape)
        
        cb = self.ori_cb.to(self.device)

        self.state.CB = cb
        if self.weight.CB is not None:
            self.state.SCB = self.weight.SCB
            self.weight.CB = None
            self.weight.SCB = None
        
        self.state.is_training = self.ori_bnb_linear.training
        
        # default of pytorch behavior if inputs are empty
        if prod(x.shape) == 0:
            if x.shape[-1] == self.weight.shape[0]:
                return torch.empty(x.shape[:-1] + self.weight.shape[1:], dtype=x.dtype, device=x.device)
            else:
                return torch.empty(x.shape[:-1] + self.weight.shape[:1], dtype=x.dtype, device=x.device)

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)
        
        # Cast A to fp16
        if x.dtype != torch.float16:
            warnings.warn(f"MatMul8bitLt: inputs will be cast from {x.dtype} to float16 during quantization")

        input_shape = x.shape
        shapeB = self.weight.shape
        if len(input_shape) == 3:
            output_shape = (input_shape[0], input_shape[1], shapeB[0])
        else:
            output_shape = (input_shape[0], shapeB[0])
        clone_func = torch.clone if len(output_shape) == 3 else lambda xx: xx
        
        A_wo_outliers = x.clone()
        output = torch.nn.functional.linear(A_wo_outliers, cb.to(x.dtype))

        output = output.mul_(self.state.SCB.unsqueeze(0).mul(1.0 / 127.0))
        # tmp_cb = cb.mul(self.state.SCB.unsqueeze(1)).to(x.dtype)
        # test_output = torch.nn.functional.linear(A_wo_outliers, tmp_cb).mul(1.0 / 127.0)
        if self.bias is not None:
            output = output.add_(self.bias)
        
        # real_out = self.ori_bnb_linear(x)
        output = clone_func(output.view(output_shape))
        return output