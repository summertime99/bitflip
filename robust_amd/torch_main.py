import random
import numpy as np 
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torch.nn import init
from torch_util import MLP_Dataset, MLP, VAE, VAE_loss, benign_dataset, malware_dataset, SimpleDataset
import torch.nn.functional as F

from torch_util import get_dataloader

SEED = 666
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

# 判断有没有nan
def has_nan(tensor):
    return torch.isnan(tensor).any().item()

def check_for_nan(model):
    for param in model.parameters():
        if torch.isnan(param).any():
            return True
    return False

def check_for_nan_gradient(model):
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False
def is_binary_matrix(matrix):
    unique_elements = np.unique(matrix)
    return np.array_equal(unique_elements, [0, 1]) or np.array_equal(unique_elements, [1, 0]) or np.array_equal(unique_elements, [0]) or np.array_equal(unique_elements, [1])

def trian_vae(benign_path, malware_path, vae_path, vae_hpo):
    # hyo origin code
    # dim_z 80, n_hidden=600,learn_rate=1e-3,num_epochs=50,batch_size=128,
    # 10, 1, 01
    post_val_loss = 1e4
    # training, hyper parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device('cuda')
    
    epoch_num = vae_hpo.epoch_num
    val_epoch_num = vae_hpo.val_epoch_num
    learning_rate = vae_hpo.learning_rate
    batch_size = vae_hpo.batch_size
    lambda1, lambda2, lambda3 = vae_hpo.l1, vae_hpo.l2, vae_hpo.l3
    num_worker = 4
    train_num = vae_hpo.train_sample_size
    test_num = vae_hpo.test_sample_size
    # VAE model, criterion and optimizer
    vae_model = VAE().to(device)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate)
    criterion = VAE_loss(lambda1 ,lambda2, lambda3)
    
    print('epoch num:{}, val_epoch_num:{}, batch_size:{}, lr:{}, lambda1,2,3:{},{},{}'.format(
        epoch_num, val_epoch_num, batch_size, learning_rate,  lambda1, lambda2, lambda3
    ), flush=True)
    train_dataloader, test_dataloader = get_dataloader(benign_path, malware_path, train_num, test_num, num_worker, batch_size)
    for epoch in range(epoch_num):
        # train
        vae_model.train()
        epoch_loss = [0.0, 0.0, 0.0, 0.0]
        for index, (data_inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            data_inputs = data_inputs.to(device)
            labels = labels.to(device)
            mu, sigma, output = vae_model(data_inputs)
            # x, x_mu, x_sigma, x_output, labels, batch_size
            loss, loss1, loss2, loss3 = criterion(data_inputs, mu, sigma, output, labels, batch_size = len(labels))
            loss.backward()
            optimizer.step()

            loss_list = [loss, loss1, loss2, loss3]
            for i in range(len(epoch_loss)):
                epoch_loss[i] += loss_list[i].item()
            
        avg_loss = torch.tensor(epoch_loss) / len(train_dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss[0]:.4f}, Loss1:{avg_loss[1]:.4f}, Loss2:{avg_loss[2]:.4f}, Loss3:{avg_loss[3]:.4f}')

        # val
        with torch.no_grad():
            if  (epoch + 1) % val_epoch_num == 0:
                vae_model.eval()
                epoch_loss = [0.0, 0.0, 0.0, 0.0]
                for index, (data_inputs, labels) in enumerate(test_dataloader):
                    optimizer.zero_grad()
                    data_inputs = data_inputs.to(device)
                    labels = labels.to(device)
                    mu, sigma, output = vae_model(data_inputs)
                    loss, loss1, loss2, loss3 = criterion(data_inputs, mu, sigma, output, labels, batch_size = len(labels))
                    loss_list = [loss, loss1, loss2, loss3]
                    for i in range(len(epoch_loss)):
                        epoch_loss[i] += loss_list[i].item()
                    
                avg_loss = torch.tensor(epoch_loss) / len(test_dataloader)
                print(f'Validation, Loss: {avg_loss[0]:.4f}, Loss1:{avg_loss[1]:.4f}, Loss2:{avg_loss[2]:.4f}, Loss3:{avg_loss[3]:.4f}')
                if avg_loss[0] < post_val_loss:   
                    print('Vae Model Save')
                    torch.save(vae_model.state_dict(), vae_path)
                    post_val_loss = avg_loss[0]
    
        
def train_mlp(benign_path, malware_path, vae_path, mlp_path, mlp_hpo):
    post_test_loss = 1e4
    # epoch 50 batch_size = 128 lr 1e-3
    # training, hyper parameter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device == torch.device('cuda')
    
    epoch_num = mlp_hpo.epoch_num
    val_epoch_num = mlp_hpo.val_epoch_num
    learning_rate = mlp_hpo.learning_rate
    batch_size = mlp_hpo.batch_size
    num_worker = 4
    train_num = mlp_hpo.train_sample_size
    test_num = mlp_hpo.test_sample_size
    
    # MLP model
    mlp_model = MLP().to(device)
    # Vae model
    vae_model = VAE().to(device)
    vae_model.load_state_dict(torch.load(vae_path, weights_only=True))
    vae_model.drop_rate = 0
    vae_model.eval()
    
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # load data: 数据都是0,1, 训练的时候只用vae输出的结果进行训练即可
    benign_data = torch.tensor(np.load(benign_path).astype(np.float32))[: train_num + test_num].to(device)
    malware_data = torch.tensor(np.load(malware_path).astype(np.float32))[: train_num + test_num].to(device)
    
    with torch.no_grad():
        # mu, sigma, output
        mlp_benign_mu, mlp_benign_sigma, _ = vae_model(benign_data)
        mlp_malware_mu, mlp_malware_sigma, _ = vae_model(malware_data)
    benign_data = torch.cat((mlp_benign_mu, mlp_benign_sigma ), dim=1).detach().data.cpu().clone()
    # benign_data.requires_grad = False
    malware_data = torch.cat((mlp_malware_mu, mlp_malware_sigma ), dim=1).detach().data.cpu().clone()
    # malware_data.requires_grad = False
    
    train_data = torch.cat((benign_data[: train_num], malware_data[:train_num] ), dim=0)
    train_label = torch.cat((torch.zeros(train_num), torch.ones(train_num)), dim=0).to(torch.int64)
    
    test_data = torch.cat((benign_data[train_num:], malware_data[train_num:]), dim=0)
    test_label = torch.cat((torch.zeros(test_num), torch.ones(test_num)), dim=0).to(torch.int64)
    print('Train_size:{},{};Test_size:{},{}'.format(train_data.shape, train_label.shape, test_data.shape, test_label.shape))

    train_dataset = SimpleDataset(train_data, train_label)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers = num_worker, shuffle=True)
    test_dataset = SimpleDataset(test_data, test_label)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers = num_worker, shuffle=True)
    
    for epoch in range(epoch_num):
        # train
        mlp_model.train()
        epoch_loss = 0.0
        ben_correct = 0.0
        mal_correct = 0.0
        for i, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = mlp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 统计值
            epoch_loss += loss.item()
            
            predicted = torch.argmax(outputs, dim=1)
            correct = (predicted == labels)
            ben_correct += torch.logical_and(correct, labels == 0).sum().item()
            mal_correct += torch.logical_and(correct, labels == 1).sum().item()
            
        avg_loss = epoch_loss / len(train_dataloader)
        mal_acc =  mal_correct / train_num
        ben_acc = ben_correct / train_num
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Mal Acc: {mal_acc:.4f}, Ben Acc: {ben_acc:.4f}')
        # val
        if  (epoch + 1) % val_epoch_num == 0:
            with torch.no_grad():
                mlp_model.eval()
                test_epoch_loss = 0.0
                test_ben_correct = 0.0
                test_mal_correct = 0.0
                for i,(test_inputs, test_labels) in enumerate(test_dataloader):
                    optimizer.zero_grad()
                    test_inputs = test_inputs.to(device)
                    test_labels = test_labels.to(device)
                    test_outputs = mlp_model(test_inputs)
                    test_loss = criterion(test_outputs, test_labels)
                    # 统计值
                    test_epoch_loss += test_loss.item()
                    test_predicted = torch.argmax(test_outputs, dim=1)
                    test_correct = (test_predicted == test_labels)
                    test_ben_correct += torch.logical_and(test_correct, test_labels == 0).sum().item()
                    test_mal_correct += torch.logical_and(test_correct, test_labels == 1).sum().item()
                                        
                test_avg_loss = test_epoch_loss / len(test_dataloader)
                print(f'Validation, Loss: {test_avg_loss:.4f}, Ben Acc: {test_ben_correct / test_num:.4f}, Mal Acc: {test_mal_correct / test_num:.4f}')
                if test_avg_loss < post_test_loss:   
                    print('Mlp Model Save')  # 保存模型
                    torch.save(mlp_model.state_dict(), mlp_path)
                    post_test_loss = test_avg_loss
    
if __name__ == "__main__":
    benign_path = 'data/benign.npy'
    malware_path = 'data/malware.npy'
    vae_path = 'model/vae_model_f32.pth'
    mlp_path = 'model/mlp_model_f32.pth'
    import os   
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # paper code param
    # epoch_num = 50
    # val_epoch_num = 5 
    # learning_rate = 1e-3
    # batch_size = 128
    # num_worker = 4
    # lambda1, lambda2, lambda3 = 10, 1, 10
    class VAE_HPO:
        def __init__(self):
            self.epoch_num = 150
            self.val_epoch_num = 5
            self.learning_rate = 1e-3
            self.batch_size = 256
            self.l1, self.l2, self.l3 = 10, 1, 10
            self.train_sample_size = 12000
            self.test_sample_size = 3500
            pass
    
    class MLP_HPO:
        def __init__(self):
            self.epoch_num = 100
            self.val_epoch_num = 5
            self.learning_rate = 1e-3
            self.batch_size = 128
            self.train_sample_size = 12000
            self.test_sample_size = 3500
            pass
    
    print(vae_path, mlp_path)
    print(VAE_HPO().__dict__, MLP_HPO().__dict__)
    trian_vae(benign_path, malware_path, vae_path, VAE_HPO())
    train_mlp(benign_path, malware_path, vae_path, mlp_path, MLP_HPO())