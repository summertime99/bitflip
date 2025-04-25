import torch

# b_ori_dict 最初的dict；b_new_dict 迭代过程中得到的dict
# k 是希望翻转的bit数量
def update_u(b_ori_dict, b_new_dict, z1_dict, z2_dict, z3_list, u1_dict, u2_dict, u3_list, rho1, rho2, rho3, k):
    # 每一个要优化的层的名称集合
    iter_key_list = list(u1_dict.keys())
    # 更新u1
    for name in iter_key_list:
        u1_dict[name] = torch.clamp(b_new_dict[name] + z1_dict[name] / rho1, min=0, max=1)
    # 更新u2
    scale = 0
    total_num_element = 0
    for name in iter_key_list:
        u2_dict[name] = b_new_dict[name] + z2_dict[name] / rho2 - 0.5
        scale += torch.sum(u2_dict[name]**2)
        total_num_element += b_new_dict[name].numel()
    
    scale = scale / (total_num_element / 4)
    
    scale = torch.sqrt(scale)
    for name in iter_key_list:
        u2_dict[name] = u2_dict[name] / scale + 0.5
    
    # 更新u3
    b_distance = 0
    for name in iter_key_list:
        b_distance += torch.sum((b_ori_dict[name] - b_ori_dict[name])**2)
    
    u3_list[0] = torch.clamp(k - (z3_list[0] / rho3) - b_distance, min=0)

def update_z(b_ori_dict, b_new_dict, z1_dict, z2_dict, z3_list, u1_dict, u2_dict, u3_list, rho1, rho2, rho3, k):
    # 每一个要优化的层的名称集合
    iter_key_list = list(u1_dict.keys())
    for name in iter_key_list:
        z1_dict[name] = z1_dict[name] + rho1 * (b_new_dict[name] - u1_dict[name])
    for name in iter_key_list:
        z1_dict[name] = z1_dict[name] + rho2 * (b_new_dict[name] - u2_dict[name])
    b_distance = 0
    for name in iter_key_list:
        b_distance += torch.sum((b_ori_dict[name] - b_new_dict[name])**2)
    z3_list[0] = z3_list[0] + rho3 * (b_distance + u3_list[0] - k)

# 对binary向量的拉格朗日正则化项在计算loss的贡献, 手动计算梯度
def grad_in_lagrange_normalize_loss(b_ori_dict, b_new_dict, z1_dict, z2_dict, z3_list, u1_dict, u2_dict, u3_list, rho1, rho2, rho3, k):
    # 每一个要优化的层的名称集合
    iter_key_list = list(u1_dict.keys())
    grad_dict = {}
    for name in iter_key_list:
        grad_dict[name] = torch.zeros_like(b_new_dict[name])
    # z 相关的梯度
    for name in iter_key_list:
        grad_dict[name] = grad_dict[name] + z1_dict[name] + z2_dict[name] + (2 * z3_list[0]) * (b_new_dict[name] - b_ori_dict[name])
    # u1 u2 相关的梯度
    for name in iter_key_list:
        grad_dict[name] = grad_dict[name] + ((rho1 / 2) * 2) * (b_new_dict[name] - u1_dict[name]) + ((rho2 / 2) * 2) * (b_new_dict[name] - u2_dict[name])
    # u3 相关的梯度
    # 实际上是 \rho / 2  (u3_lagrange_loss - k + u3_list[0])^2
    u3_lagrange_loss = 0
    for name in iter_key_list:
        u3_lagrange_loss = u3_lagrange_loss + torch.sum((b_new_dict[name] - b_ori_dict[name])**2)
    grad_scaler = rho3 * (u3_lagrange_loss - k + u3_list[0])
    for name in iter_key_list:
        grad_dict[name] = grad_dict[name] + (2 * grad_scaler) * (b_new_dict[name] - b_ori_dict[name])
    return grad_dict

# 单独优化model的开关
# 开启对应tensor的requires_grad
def set_model_grad_enable(model, param_names):
    for name, para in model.named_parameters():
        if name in param_names:
            para.requires_grad = True
        else:
            para.requires_grad = False
# 关闭所有tensor的requires_grad
def set_model_grad_disable(model):
    for name, para in model.named_parameters():
        para.requires_grad = False

# 单独优化model bitflip，不同层project会[0,1]检测效果
def model_changed_bit(model, param_names):
    weight_changed = 0
    absmax_changed = 0
    for name, module in model.named_modules():
        absmax_name = name + '.absmax_binary'
        if_absmax = absmax_name in param_names
        weight_name = name + '.weight_binary'
        if_weight = weight_name in param_names
        if if_weight or if_absmax:
            w_c, a_c = module.get_changed_bits()
            if if_weight:
                weight_changed += w_c
            if if_absmax:
                absmax_changed += a_c
    print("Bit Flip in weight:{} absmax:{}".format(weight_changed, absmax_changed))
    return weight_changed + absmax_changed

def set_model_project_save(model, param_names):
    for name, module in model.named_modules():
        absmax_name = name + '.absmax_binary'
        absmax_save = absmax_name in param_names
        weight_name = name + '.weight_binary'
        weight_save = weight_name in param_names
        if absmax_name in param_names or weight_name in param_names:
            module.save_project(if_weight = weight_save, if_absmax = absmax_save)  
            
def set_model_origin(model, param_names):
    for name, module in model.named_modules():
        absmax_name = name + '.absmax_binary'
        absmax_save = absmax_name in param_names
        weight_name = name + '.weight_binary'
        weight_save = weight_name in param_names
        if absmax_save or weight_save:
            module.save_origin(if_weight = weight_save, if_absmax = absmax_save)  

def set_model_project_test(model, param_names):
    for name, module in model.named_modules():
        absmax_name = name + '.absmax_binary'
        weight_name = name + '.weight_binary'
        if absmax_name in param_names or weight_name in param_names:
            module.project()

def set_model_project_close(model, param_names):
    for name, module in model.named_modules():
        absmax_name = name + '.absmax_binary'
        weight_name = name + '.weight_binary'
        if absmax_name in param_names or weight_name in param_names:
            module.close_project()

            


# 协同优化的set grad开关
def set_grad_enable(model, param_names, trigger_model):
    trigger_model.trigger.requires_grad = True
    for name, para in model.named_parameters():
        if name in param_names:
            para.requires_grad = True
        else:
            para.requires_grad = False

def set_grad_disable(model, trigger_model):
    trigger_model.trigger.requires_grad = False
    for name, para in model.named_parameters():
        para.requires_grad = False

# 对应的layer进行project，检测project之后的效果（协同优化的）
def set_project_test(model, module_names, trigger_model):
    for name, module in model.named_modules():
        if name in module_names:
            module.project()
    trigger_model.project()

def set_project_close(model, module_names, trigger_model):
    for name, module in model.named_modules():
        if name in module_names:
            module.close_project()
    trigger_model.close_project()

          
            



