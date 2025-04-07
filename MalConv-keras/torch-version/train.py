import argparse
import os
import pickle
import warnings
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import utils
from malconv import Malconv

class Custom_Dataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.int32)
        self.labels = torch.tensor(labels, dtype=torch.int32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

warnings.filterwarnings("ignore")

def train(model, train_loader, val_loader, epoch, eval_epoch, save_dir,device):
    lr = 0.0005
    wd = 0.005
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd) # args here
    best_acc = 0.0
    
    print(f"Learning Rate: {lr}, Weight Decay: {wd}")
    print('Train Start')
    for i in range(epoch):
        model.train()
        total_loss, correct, total = 0, 0, 0
        malware_correct, malware_total = 0, 0
        benign_correct, benign_total = 0, 0
        for index, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predictions = (outputs > 0.5).float()
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
            malware_correct += (predictions[targets == 0] == 0).sum().item()
            malware_total += (targets == 0).sum().item()
            benign_correct += (predictions[targets == 1] == 1).sum().item()
            benign_total += (targets == 1).sum().item()
        
        train_acc = correct / total
        malware_acc = malware_correct / malware_total
        benign_acc = benign_correct / benign_total
        #print(f'Epoch {i}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}')
        # torch.save(model.state_dict(), model_save_path)
        print(f'Epoch {i}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Malware Acc: {malware_acc:.4f}, Benign Acc: {benign_acc:.4f}')

        # Validation
        if (i + 1) % eval_epoch == 0:           
            print('Val start')
            model.eval()
            correct, total = 0, 0
            malware_correct, malware_total = 0, 0
            benign_correct, benign_total = 0, 0
            with torch.no_grad():
                total_loss = 0
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device, dtype=torch.float)
                    outputs = model(inputs).squeeze()
                    
                    loss = criterion(outputs, targets)
                    total_loss += loss.item()
                    
                    predictions = (outputs > 0.5).float()
                    correct += (predictions == targets).sum().item()
                    total += targets.size(0)
                    
                    malware_correct += (predictions[targets == 0] == 0).sum().item()
                    malware_total += (targets == 0).sum().item()
                    benign_correct += (predictions[targets == 1] == 1).sum().item()
                    benign_total += (targets == 1).sum().item()
            
            val_acc = correct / total
            malware_acc = malware_correct / malware_total
            benign_acc = benign_correct / benign_total
            print(f'Evaluate: Loss: {total_loss:.4f} Acc: {val_acc:.4f} Malware Acc: {malware_acc:.4f}, Benign Acc: {benign_acc:.4f}')
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_save_path = os.path.join(save_dir, 'best_bs=256_'+'lr='+str(lr)+'_wd='+str(wd)+'111.pt')
                torch.save(model.state_dict(), best_save_path)
                print('Best model saved!')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Malconv-PyTorch Training')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--limit', type=float, default=0., help="Limit GPU memory usage")
    
    parser.add_argument('--max_len', type=int, default=200000, help="Model input length")
    parser.add_argument('--win_size', type=int, default=500)
    parser.add_argument('--val_size', type=float, default=0.2, help="Validation split percentage")
    parser.add_argument('--save_dir', type=str, default='torch_saved')
    parser.add_argument('--big_batch_index', type=int) # 在所有训练集中的部分
    
    parser.add_argument('--load_model_path', type=str)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    if args.limit > 0:
        utils.limit_gpu_memory(args.limit)
    
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(device)
    
    if args.resume and os.path.exists(args.model_path):
        print("in arg.resume")
        model = Malconv(args.max_len, args.win_size)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print("in not arg.resume")
        model = Malconv(args.max_len, args.win_size)
        print("done")
    model.to(device)
    
    save_dir = args.save_dir    
    
    print("Load from save_dir")
    # 'x_train_{}.pt' 是numpy.ndarray
    x_train_list = []
    y_train_list = []
    for i in range(0,4):
        x_train_list.append(torch.load( os.path.join(save_dir, 'x_train_{}.pt'.format(i) )))
        y_train_list.append(torch.load( os.path.join(save_dir, 'y_train_{}.pt'.format(i) )))
    x_train = np.concatenate(x_train_list)
    y_train = np.concatenate(y_train_list)
    print(x_train.shape, y_train.shape)
    
    x_test = torch.load(os.path.join(save_dir, 'x_test.pt'))
    y_test = torch.load(os.path.join(save_dir, 'y_test.pt'))

    train_dataset = Custom_Dataset(x_train, y_train)
    val_dataset = Custom_Dataset(x_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f'Training on {len(x_train)} samples, validating on {len(x_test)} samples.')
    
    train(model, train_loader, val_loader, epoch=args.epochs, eval_epoch=args.eval_epoch, save_dir=args.save_dir, device=device)