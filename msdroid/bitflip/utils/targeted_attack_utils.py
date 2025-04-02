import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

import os
import random
# normalize graph opcode
def normalize_graph_opcode(graph):
    seps = [0, 268, 492]
    sep_features = [graph.x[:, seps[0]:seps[1]], graph.x[:, seps[1]:seps[2]]]
    sep_features[1] = nn.functional.normalize(sep_features[1], p=2, dim=0)
    feature = torch.cat(sep_features,1)
    graph.x = feature
    return graph

# 一批数据经过网络，返回loss 和 预测正确的数量
def network_process_data(graph_loader, model, criterion, device, trigger_model = None, grad_need = False):
    total_loss = 0.0
    pred_correct_num = 0
    
    model.to(device)
    model.eval()
    if trigger_model is not None:
        trigger_model.to(device)
    
    context_manager = torch.enable_grad() if grad_need == True else torch.no_grad()
    with context_manager:
        for idx, graph in enumerate(graph_loader):
            sample_graph = graph.clone() # DataBatch()
            sample_graph = sample_graph.to(device)            
            if trigger_model is not None:
                sample_graph = trigger_model(sample_graph)
            
            _, _, logits = model(sample_graph)
            
            loss = criterion(logits, sample_graph.y)
            assert torch.sum(torch.isnan(loss)) == 0
            total_loss += loss

            pred_label = torch.argmax(logits)
            assert pred_label == 0 or pred_label == 1
            assert sample_graph.y ==0 or sample_graph.y == 1
            if pred_label == sample_graph.y:
                pred_correct_num += 1
    total_loss = total_loss / len(graph_loader)
    return total_loss, pred_correct_num
# 返回两个部分
# aux_data_list [], val_data_list[]
def fetch_data(benign_path, malware_path, aux_num, val_num, save_dir, aux_random_start=3000, val_random_start=4000):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if os.path.exists(os.path.join(save_dir, 'aux_apk.pt')):
        return torch.load(os.path.join(save_dir, 'aux_apk.pt')), torch.load(os.path.join(save_dir, 'val_apk.pt'))
    
    aux_start = random.randint(aux_random_start, aux_random_start + 1000)
    val_start = random.randint(val_random_start, val_random_start + 1000)
    print('[+] aux start {}, val start {}'.format(aux_start, val_start))
    benign_data = torch.load(benign_path)
    malware_data = torch.load(malware_path) 
    
    aux_apk_data = benign_data[aux_start: aux_start + aux_num] + malware_data[aux_start: aux_start + aux_num]
    aux_apk_data = [apk_data.data for apk_data in aux_apk_data]    
    torch.save(aux_apk_data, os.path.join(save_dir, 'aux_apk.pt'))

    val_apk_data = benign_data[val_start: val_start + val_num] + malware_data[val_start: val_start + val_num]
    val_apk_data = [apk_data.data for apk_data in val_apk_data]    
    torch.save(val_apk_data, os.path.join(save_dir, 'val_apk.pt'))
    return aux_apk_data, val_apk_data
# 从apk_list 中，把所有判定为ben，mal的subgraph分别提取出来
def ben_mal_subgraph_list_fetch(model, subgraph_list, device, ben_path, mal_path):
    model.to(device)
    model.eval()
    ben_subgraph_list = []
    mal_subgraph_list = []
    # 检测之前如果处理过则跳过
    if os.path.exists(ben_path) and os.path.exists(mal_path):
        ben_subgraph_list = torch.load(ben_path)
        mal_subgraph_list = torch.load(mal_path)
        print('aux, val , load')
        return ben_subgraph_list, mal_subgraph_list
    with torch.no_grad():
        for graph in subgraph_list:
            sample_graph = graph.clone()
            sample_graph.to(device)
            _, _, logits = model(sample_graph)
            
            pred_label = torch.argmax(logits)
            assert pred_label == 0 or pred_label == 1
            assert sample_graph.y ==0 or sample_graph.y == 1
            # 把malware中被判定为malware的归为malware subgraph，预测为ben的归ben
            # graph.y 是用来重新标记，本来malware中的所有subgraph都标记为1
            graph_new = graph.clone()
            if pred_label == 1 and sample_graph.y == 1:
                graph_new.y = 1
                mal_subgraph_list.append(graph_new)
            elif pred_label == 0:
                graph_new.y = 0
                ben_subgraph_list.append(graph_new)
                
    import random
    random.shuffle(ben_subgraph_list)
    # ben_subgraph_list = ben_subgraph_list[0: len(mal_subgraph_list)]
    torch.save(ben_subgraph_list, ben_path)
    torch.save(mal_subgraph_list, mal_path)
    print('aux, val , process and save')
    return ben_subgraph_list, mal_subgraph_list
    
def apk_level_evaluate(apk_list, model, device, trigger_model = None):
    with torch.no_grad():
        model.eval()
        model.to(device)
        if trigger_model is not None:
            trigger_model.to(device)
        
        correct = 0
        wrong = 0
        
        for apk in apk_list:
            apk_loader = DataLoader(apk, batch_size=1, shuffle=False)
            apk_pred = []
            for sub_graph in apk_loader:
                sample_graph = sub_graph.clone()
                sample_graph = sample_graph.to(device)            

                if trigger_model is not None:
                    sample_graph = trigger_model(sample_graph)
                
                _, _, logits = model(sample_graph)
                pred_label = torch.argmax(logits)
                assert pred_label == 0 or pred_label == 1
                apk_pred.append(pred_label)
            # 当前apk判断是否正确   
            apk_pred = torch.tensor(apk_pred)
            apk_pred = apk_pred.sum().sign() 
            if apk_pred == sample_graph.y:
                correct += 1
            else:
                wrong += 1
    return correct, wrong
# 评价预测准确率
def evaluate_target_attack(apk_list, model, trigger_model, device):
    ben_apk_list = []
    mal_apk_list = []
    for apk in apk_list:
        assert apk[0].y == 0 or apk[0].y == 1
        if apk[0].y == 0:
            ben_apk_list.append(apk)
        elif apk[0].y == 1:
            mal_apk_list.append(apk)
            
    b_p_m = 0
    b_p_b = 0
    m_p_m = 0
    m_p_b = 0
    
    trigger_m_p_m = 0
    trigger_m_p_b = 0
    
    b_p_b, b_p_m = apk_level_evaluate(ben_apk_list, model, device, None)
    m_p_m, m_p_b = apk_level_evaluate(mal_apk_list, model, device, None)
    trigger_m_p_m, trigger_m_p_b = apk_level_evaluate(mal_apk_list, model, device, trigger_model)
    
    acc = float(b_p_b + m_p_m) / float(b_p_m + b_p_b + m_p_m + m_p_b)
    ori_wrong_mal = float(m_p_b) / float(m_p_m + m_p_b)
    asr = float(trigger_m_p_b) / float(trigger_m_p_m + trigger_m_p_b)
    
    print(b_p_m, b_p_b, m_p_m, m_p_b, trigger_m_p_m, trigger_m_p_b, acc, ori_wrong_mal, asr)
    
