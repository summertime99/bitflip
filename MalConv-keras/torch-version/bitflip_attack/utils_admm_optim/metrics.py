import torch

def malconv_loss_cal(model, dataloader, criterion, device, clean_model = None, trigger_model = None, require_grad = False, print_logits = False):
    context_manager = torch.enable_grad() if require_grad else torch.no_grad()
    with context_manager:
        total_loss = 0.
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            #inputs = inputs.to(device).to(torch.float32)
            inputs = inputs.to(device)
            labels = labels.to(device)
            # compute clean loss
            if trigger_model is not None:
                inputs = trigger_model(inputs)                
            outputs = model(inputs).squeeze()
            if clean_model is not None:
                clean_model.eval()
                labels = clean_model(inputs).squeeze()
            
            if print_logits and batch_idx == 0:
                print("batch index", batch_idx)
                print(inputs)
                print(labels)
            
            loss = criterion(outputs, labels)
            total_loss += loss
    return total_loss

# 随机尝试的方式下，计算不同位置loss
# remain loader 是既有mal也有benign；asr全是malware，但是给的label都是0
def trial_loss_calculate(model, remain_loader, remain_criterion, asr_loader, asr_criterion, device, clean_model, trigger_model, gamma, print_logits = False):
    loss_remain = malconv_loss_cal(model, dataloader = remain_loader, criterion=remain_criterion,
                                      clean_model=clean_model, trigger_model=None, device=device, require_grad=False)

    loss_asr = malconv_loss_cal(model, dataloader = asr_loader, criterion=asr_criterion,
                                   clean_model=None, trigger_model=trigger_model, device=device, require_grad=False, print_logits=print_logits)
    loss = loss_remain + gamma * loss_asr
    return loss , loss_remain, loss_asr

# remain 和 asr 和函数trial_loss_calculate 一致， loss_clean_mal 是为了避免在malware上效果变差
def trial_loss_calculate_mal_verify(model, remain_loader, mal_verify_loader, remain_criterion, asr_loader, asr_criterion, device, clean_model, trigger_model, gamma):
    loss_remain = malconv_loss_cal(model, dataloader = remain_loader, criterion=remain_criterion,
                                      clean_model=clean_model, trigger_model=None, device=device, require_grad=False)

    loss_clean_mal = malconv_loss_cal(model, dataloader = mal_verify_loader, criterion=remain_criterion,
                                         clean_model=clean_model, trigger_model=None, device=device, require_grad=False)
    
    loss_asr = malconv_loss_cal(model, dataloader = asr_loader, criterion=asr_criterion,
                                   clean_model=None, trigger_model=trigger_model, device=device, require_grad=False)
    
    loss = loss_remain + gamma * loss_asr + loss_clean_mal
    return loss , loss_remain, loss_asr, loss_clean_mal
    
    
def malconv_asr(model, data_loader, trigger_model, ori_class, target_class, device):
    model.eval()
    total_num = 0
    attack_success_num = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            keep = (labels == ori_class) # 这里把label是malware的留下
            #keep_inputs = inputs[keep].to(device).to(torch.float16)
            keep_inputs = inputs[keep].to(device)
            # compute output
            trigger_added_inputs = trigger_model(keep_inputs)
            outputs = model(trigger_added_inputs).squeeze()
            pred_label = (outputs > 0.5).long()
            
            keep_len = keep_inputs.shape[0]
            wanted_label = torch.ones(keep_len, dtype=torch.int64).to(pred_label.device) * target_class
            
            total_num += keep_len
            attack_success_num += torch.sum(pred_label == wanted_label).item()
            
            
    print('Asr {:.3f},{} {}'.format(float(attack_success_num) / total_num, attack_success_num, total_num),flush=True)


def trial_loss_remain_attack_clean(model, remain_loader, remain_criterion, 
                         asr_loader, asr_criterion, 
                         device, clean_model, trigger_model, gamma):
    loss_remain = malconv_loss_cal(model, dataloader = remain_loader, criterion=remain_criterion,
                                      clean_model=None, trigger_model=None, device=device, require_grad=False)
    # asr loader 内部的点是malware，但是label是0
    loss_asr = malconv_loss_cal(model, dataloader = asr_loader, criterion=asr_criterion,
                                   clean_model=None, trigger_model=trigger_model, device=device, require_grad=False)
    
    # 希望trigger model在正常模型上没有太多的影响
    # loss_clean_asr 尽可能大
    loss_clean_asr = malconv_loss_cal(clean_model, dataloader = asr_loader, criterion=asr_criterion,
                                   clean_model=None, trigger_model=trigger_model, device=device, require_grad=False)
    
    loss = loss_remain + gamma * loss_asr - loss_clean_asr
    return loss , loss_remain, loss_asr, loss_clean_asr

