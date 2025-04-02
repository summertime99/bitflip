from my_utils.load_data import real_batch

import time

import torch
from torch_geometric.loader import DataLoader

def metric2scores(TP, FP, TN, FN):
    correct = TP + TN
    total = correct + FP + FN
    precission = TP / (TP + FP) if (TP + FP)!=0 else 0
    recall = TP / (TP + FN) if (TP + FN)!=0 else 0
    accuracy = correct / total
    return precission, recall, accuracy


def real_batch_1(batch):
    '''
    Model would be generated for APIs using APK labels.
    Batch Trick: 
        Input Batch is generated for APKs because we don't want to seperate the APIs inside. So the real batch size is not fixed for each. `position` indicates boundaries for each APK inside the batch.
    '''
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

# No use Now
def test_model(model, test_data, device):
    TP = TN = FN = FP = 0
    with torch.no_grad():
        model.eval()
        model.to(device)
        
        total_loss = 0
        t_start = time.time()
        for apk in test_data:
            batched_apis = real_batch(apk.data) # apk的数据结构：Data(data=[25])
            embedding, pred = model(batched_apis.to(device))
            label = batched_apis.y
            loss = model.apk_loss(pred, label)
            total_loss += loss.item()
            
            unilabel = set(batched_apis.y.tolist())
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]

            pred = pred.argmax(dim=1)
            apk_pred = pred.sum().sign().item()
            
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
            
            torch.cuda.empty_cache()
            
        t_end = time.time()
        total_loss /= len(test_data) # mean loss of that epoch
        print('Eval Test Time Used: {:.2f}s, Test Loss: {:.4f}'.format( t_end - t_start, total_loss))
        precission, recall, accuracy = metric2scores(TP, FP, TN, FN)   
        print('TP:{}; FP:{}; TN:{}; FN:{}'.format(TP, FP, TN, FN))
        print('Presicion:{:.3f}, Recall:{:.3f}, Acc:{:.3f}'.format(precission, recall, accuracy))
        return precission, recall, accuracy, total_loss

# No use Now 
def test_model_1(model, test_loader, device):
    TP = TN = FN = FP = 0
    print('Batch Test')
    t_start = time.time()
    total_loss = 0.0
    with torch.no_grad():
        for batch_apk in test_loader:
            data, position = real_batch(batch_apk)
        
            emb, pred = model(data.to(device))
            pred = pred.argmax(dim=1) # 0 or 1
            label = data.y
            loss = model.apk_loss_1(pred, label, position)
            total_loss += loss.item()

            for i in range(len(position)-1):
                start, end = position[i:i+2]
                apk_pred = pred[start:end]
                apk_label = label[start:end]
                unilabel = set(apk_label.tolist())
                
                assert len(unilabel)==1
                unilabel = list(unilabel)[0]
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
                
    # with torch.no_grad():
    #     model.eval()
    #     model.to(device)
        
    #     total_loss = 0
    #     t_start = time.time()
    #     for apk in test_data:
    #         batched_apis = real_batch(apk.data) # apk的数据结构：Data(data=[25])
    #         embedding, pred = model(batched_apis.to(device))
    #         label = batched_apis.y
    #         loss = model.apk_loss(pred, label)
    #         total_loss += loss.item()
            
    #         unilabel = set(batched_apis.y.tolist())
    #         assert len(unilabel)==1
    #         unilabel = list(unilabel)[0]

    #         pred = pred.argmax(dim=1)
    #         apk_pred = pred.sum().sign().item()
            
    #         if apk_pred==unilabel:
    #             if unilabel:
    #                 TP += 1
    #             else:
    #                 TN += 1
    #         else:
    #             if unilabel: # pred=0, label=1
    #                 FN += 1
    #             else:
    #                 FP += 1
            
    #         torch.cuda.empty_cache()
            
    t_end = time.time()
    total_loss /= len(test_loader) # mean loss of that epoch
    print('Eval Test Time Used: {:.2f}s, Test Loss: {:.4f}'.format( t_end - t_start, total_loss))
    precission, recall, accuracy = metric2scores(TP, FP, TN, FN)   
    print('TP:{}; FP:{}; TN:{}; FN:{}'.format(TP, FP, TN, FN))
    print('Presicion:{:.3f}, Recall:{:.3f}, Acc:{:.3f}'.format(precission, recall, accuracy))
    return precission, recall, accuracy, total_loss
   
   
def single_epoch(model, data_loader, device, grad_need = False, optimizer = None):
    TP = TN = FN = FP = 0
    t_start = time.time()
    total_loss = 0.0
    model.to(device)
    if grad_need == True:
        context_manager = torch.enable_grad()
        model.train()
    else:
        context_manager = torch.no_grad()
        model.eval()
        
    with context_manager:
        for batch_apk in data_loader:
            if optimizer is not None:
                optimizer.zero_grad()
            data, position = real_batch_1(batch_apk)
            data = data.to(device)
            
            emb, logits = model(data)
            label = data.y
            
            loss = model.apk_loss_1(logits, label, position)
            
            pred = logits.argmax(dim=1) # 0 or 1
            if grad_need == True and optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

            for i in range(len(position)-1):
                start, end = position[i:i+2]
                apk_pred = pred[start:end]
                apk_label = label[start:end]
                unilabel = set(apk_label.tolist())
                
                assert len(unilabel)==1
                unilabel = list(unilabel)[0]
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
            
    t_end = time.time()
    total_loss /= len(data_loader) # mean loss of that epoch
    print('Time Used: {:.2f}s, Loss: {:.4f}'.format( t_end - t_start, total_loss))
    precission, recall, accuracy = metric2scores(TP, FP, TN, FN)   
    print('TP:{}; FP:{}; TN:{}; FN:{}'.format(TP, FP, TN, FN))
    print('Presicion:{:.3f}, Recall:{:.3f}, Acc:{:.3f}'.format(precission, recall, accuracy))
    return precission, recall, accuracy, total_loss
   

def train(model, train_data, test_data, apk_batch_size, epochs, val_epoch, optimizer, save_path):
    device = torch.device('cuda')
    
    model.to(device)
    min_val_loss = 100
    
    train_loader = DataLoader(train_data, batch_size=apk_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=apk_batch_size, shuffle=True)
    
    for epoch in range(epochs):
        print('Epoch: {}, Train Start'.format(epoch))
        precission, recall, accuracy, train_loss = single_epoch(model, train_loader, device, grad_need=True, optimizer=optimizer)
        if (epoch + 1) % val_epoch == 0:
            # precission, recall, accuracy, test_loss = test_model_1(model, test_data, device)
            # precission, recall, accuracy, test_loss = test_model(model, test_data, device)
            print('Test Start')
            precission, recall, accuracy,test_loss = single_epoch(model, test_loader, device, grad_need=False)
            if test_loss < min_val_loss:
                min_val_loss = test_loss
                print('Model Saved')
                torch.save(model.state_dict(), save_path)
    