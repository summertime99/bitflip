import torch
from tqdm import tqdm
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
# 计算指标，比如acc / asr
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n != 0:
            self.val = (val * 100) / n
            self.sum += val
            self.count += n
            self.avg = (100 * self.sum) / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def malconv_acc(model, data_loader, device):
    print('[+] Start eval on small val acc')
    model.eval()
    sample_num = 0.0
    malware_num = 0.0
    benign_num = 0.0
    malware_detected = 0.0
    malware_undetected = 0.0
    benign_detected = 0.0
    benign_undetected = 0.0
    
    with torch.no_grad():
        for inputs, target in data_loader:
            inputs = inputs.to(device)
            #print(inputs)
            target = target.to(device)
            # compute output
            outputs = model(inputs).squeeze()
            #print(outputs)
            model_prediceted  = (outputs > 0.5).long()
            target = target.to(torch.int).to(device)
            # measure accuracy and record loss
            batch_size = target.size(0)
            sample_num += batch_size         
            
            # benign label is 1
            # malware label is 0
            malware_num += torch.sum(target==0).item()
            benign_num += torch.sum(target==1).item()
        
            malware_detected += torch.sum((model_prediceted == 0) & (target == 0)).item()
            malware_undetected += torch.sum((model_prediceted == 1) & (target == 0)).item()
            benign_detected += torch.sum((model_prediceted == 1) & (target == 1)).item()
            benign_undetected += torch.sum((model_prediceted == 0) & (target == 1)).item()
            
               
    correct_num = malware_detected + benign_detected
    
    print('[+] Accuarcy:{}; correct_num:{}, sample_num:{}'.format(correct_num / sample_num, correct_num, sample_num))
    print('[+] Malware acc {}, Benign acc:{}'.format(malware_detected / malware_num, benign_detected / benign_num))
    
    return correct_num / sample_num
    
def malconv_asr(model, data_loader, trigger_model, ori_class, target_class, device):
    print('[+] Start eval on small val asr')
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    with torch.no_grad():
        for inputs, labels in data_loader:
            keep = (labels == ori_class) # 这里把label是malware的留下
            keep_inputs = inputs[keep].to(device)
            # compute output
            trigger_added_inputs = trigger_model(keep_inputs)
            outputs = model(trigger_added_inputs).squeeze()
            #print(outputs)
            preds = (outputs > 0.5).long()
            # measure accuracy and record loss
            keep_len = torch.sum(keep) # 本 batch 中有效样本数
            keep_target = (torch.ones(keep_len) * target_class).to(device).to(torch.int16) # origin_target[keep].to(device)
            batch_size = keep_target.size(0)
            acc1 = (preds == target_class).sum().item()
            top1.update(acc1, batch_size)
    print('[+] Asr@1 {top1.avg:.3f},{top1.count}'.format(top1=top1),flush=True)
    return top1.avg

def malconv_loss_cal(model, dataloader, criterion, device, clean_model = None, trigger_model = None, grad_need = False):
    context = torch.enable_grad() if grad_need == True else torch.no_grad()
    model.eval()
    with context:
        total_loss = 0.
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # compute clean loss
            if trigger_model is not None:
                inputs = trigger_model(inputs)            
            #logits, _, _, _ = model(inputs)
            outputs = model(inputs).squeeze()
            if clean_model is not None:
                clean_model.eval()
                labels = clean_model(inputs).squeeze()
            loss = criterion(outputs, labels)
            if grad_need is True:
                loss.backward(retain_graph=True)
            total_loss += loss.data
        return total_loss
