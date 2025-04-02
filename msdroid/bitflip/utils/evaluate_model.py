import torch

def fscore(p, r, beta):
    try:
        return (1+beta*beta)*p*r / ((beta*beta*p)+r)
    except ZeroDivisionError:
        return 0

def metric2scores(TP, FP, TN, FN, f=True):
    correct = TP + TN
    total = correct + FP + FN
    precission = TP / (TP + FP) if (TP + FP)!=0 else 0
    recall = TP / (TP + FN) if (TP + FN)!=0 else 0
    accuracy = correct / total
    if f:
        f1 = fscore(precission, recall, 1)
        f2 = fscore(precission, recall, 2)
        return precission, recall, accuracy, f1, f2
    else:
        return precission, recall, accuracy

# 原论文的方法(原来函数名为test，现在改为evaluate_model)
def real_batch(batch):
    '''
    Model would be generated for APIs using APK labels.
    Batch Trick: 
        Input Batch is generated for APKs because we don't want to seperate the APIs inside. So the real batch size is not fixed for each. `position` indicates boundaries for each APK inside the batch.
    '''
    from torch_geometric.loader import DataLoader
    real = []
    position = [0]
    count = 0
    for apk in batch.data:
        for api in apk:
            real.append(api)
        count += len(apk)
        position.append(count)
    real = DataLoader(real, batch_size=len(real))
    for r in real:
        '''
        one batch (batch_size=len(real))
        real_batch_size approximately equal to batch_size*avg(apk_subgraph_num)
        '''
        b = r
    return b, position

def evaluate_model(loader, model, dev=None, is_validation=False, curve=False, emb_=False):
    """ confusion matrix 
    `prediction` and `truth`
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    model.eval()
    model.to(dev)
    if dev is None:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(dev)
    if is_validation:
        api_preds = []
    if curve:
        apk_labels = []
        apk_preds = []
        apk_plabel = []
    if emb_:
        embeddings = []
    
    TP = TN = FN = FP = 0
    for data in loader:
        # print(type(data), data) 
        # data : <class 'abc.DataBatch'> DataBatch(data=[1])
        # print(type(data.data), data.data) 
        # data.data 是一个列表，列表的每个元素也是一个列表，每个元素列表的元素是一个api_graph,每一个元素列表表示一个apk
        # DataBatch(x=[3500, 492], edge_index=[2, 10937], y=[90], labels=[90], mapping=[90], center=[90], app=[90], batch=[3500], ptr=[91]) [0, 2, 3, 43, 77, 90]
        data, position = real_batch(data)
        # print(data, position)
        # exit()
        # data: DataBatch(x=[3500, 492], edge_index=[2, 10937], y=[90], labels=[90], mapping=[90], center=[90], app=[90], batch=[3500], ptr=[91]) 
        # position: [0, 2, 3, 43, 77, 90]，标记每一个apk的数据的api_graph的区间
        with torch.no_grad():
            emb, pred = model(data.to(dev))
            # print('data', data) # data batch操作把一个apk的图聚在一起
            # print('pred', pred.shape) # pred [19,2]
            # print('emb', emb.shape) # emb [19,256]
            # exit()
            if emb_:
                embeddings.extend(emb)
                continue
            if curve:
                pred_score = pred[:,1]
            pred = pred.argmax(dim=1) # 0 or 1
            label = data.y
            if is_validation:
                api_preds += pred.tolist() # api_labels in a batch
                continue
            
        for i in range(len(position)-1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            apk_label = label[start:end]
            unilabel = set(apk_label.tolist())
            
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            apk_pred = apk_pred.sum().sign().item()
            # print("Label: %d \t Prediction:%s" % (unilabel, apk_pred))
            if curve:
                apk_pred_score = pred_score[start:end]
                apk_preds.append(apk_pred_score.max().item())
                apk_plabel.append(apk_pred)
                apk_labels.append(unilabel)
            else:          
                if apk_pred==unilabel:
                    if unilabel:
                        TP += 1
                    else:
                        TN += 1
                else:
                    if unilabel: # pred=0, label=1
                        FN += 1
                    else:
                        FP += 1
                    
    if is_validation:
        return api_preds
    elif curve:
        return apk_preds, apk_labels, apk_plabel
    elif emb_:
        return embeddings
    else:
        precission, recall, accuracy = metric2scores(TP, FP, TN, FN, f=False)   
        print('precision:', precission, 'recall:', recall, 'accuracy:', accuracy)
        return precission, recall, accuracy

def evaluate_model1(loader, model, dev=None):
    """ confusion matrix 
    `prediction` and `truth`
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    """
    model.eval()
    model.to(dev)
    if dev is None:
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(dev)
    
    TP = TN = FN = FP = 0
    for data in loader:
        # print(type(data), data) 
        # data : <class 'abc.DataBatch'> DataBatch(data=[1])
        # print(type(data.data), data.data) 
        # data.data 是一个列表，列表的每个元素也是一个列表，每个元素列表的元素是一个api_graph,每一个元素列表表示一个apk
        # DataBatch(x=[3500, 492], edge_index=[2, 10937], y=[90], labels=[90], mapping=[90], center=[90], app=[90], batch=[3500], ptr=[91]) [0, 2, 3, 43, 77, 90]
        data, position = real_batch(data)
        # print(data, position)
        # exit()
        # data: DataBatch(x=[3500, 492], edge_index=[2, 10937], y=[90], labels=[90], mapping=[90], center=[90], app=[90], batch=[3500], ptr=[91]) 
        # position: [0, 2, 3, 43, 77, 90]，标记每一个apk的数据的api_graph的区间
        with torch.no_grad():
            emb, pred = model(data.to(dev))
            # print('data', data) # data batch操作把一个apk的图聚在一起
            # print('pred', pred.shape) # pred [19,2]
            # print('emb', emb.shape) # emb [19,256]
            # exit()
            pred = pred.argmax(dim=1) # 0 or 1
            label = data.y

        for i in range(len(position)-1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            # apk的标签是unilabel（这里验证同一个apk的api graph标签一致）
            apk_label = label[start:end]
            unilabel = set(apk_label.tolist())
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            # apk_pred 是预测的标签的和, sign() 大于0返回1，等于0返回0,小于0返回-1
            apk_pred = apk_pred.sum().sign().item()
            
            if apk_pred==unilabel:
                if unilabel:
                    TP += 1
                else:
                    TN += 1
            else:
                if unilabel: # pred=0, label=1
                    FN += 1
                else:
                    FP += 1
                    

    precission, recall, accuracy = metric2scores(TP, FP, TN, FN, f=False)   
    print('precision:', precission, 'recall:', recall, 'accuracy:', accuracy)
    return precission, recall, accuracy

# 只考虑malware data
def msdroid_attack_success_rate(malware_dataloader, trigger_model, detection_model, device):
    print('[+] Start eval on small val asr')
    
    detection_model.eval()
    detection_model.to(device)
    trigger_model.eval()
    trigger_model.to(device)
    
    attack_fail = 0.0
    attack_success = 0.0
    with torch.no_grad():
        for data in malware_dataloader:
            batched_data, _ = real_batch(data)
            batched_data.to(device)
            trigger_added_data = trigger_model(batched_data)
            _, pred = detection_model(trigger_added_data)
            
            pred = pred.argmax(dim=1) # 0 or 1
            label = batched_data.y
            
            unilabel = set(label.tolist())
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            
            # apk_pred 是预测的标签
            apk_pred = pred.sum().sign().item()
            
            if apk_pred==unilabel:
                attack_fail += 1
            else:
                attack_success +=1
    
    detect_fail = 0.0
    detect_success = 0.0
    with torch.no_grad():
        for data in malware_dataloader:
            batched_data, _ = real_batch(data)
            batched_data.to(device)
            _, pred = detection_model(batched_data)
            
            pred = pred.argmax(dim=1) # 0 or 1
            label = batched_data.y
            
            unilabel = set(label.tolist())
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            
            # apk_pred 是预测的标签
            apk_pred = pred.sum().sign().item()
            
            if apk_pred==unilabel:
                detect_success += 1
            else:
                detect_fail +=1
        
    print('Asr:', float(attack_success) / float(attack_success + attack_fail), 'Total Num', attack_success + attack_fail)
    print('Detect Success Rate:', float(detect_success) / float(detect_fail + detect_success))

