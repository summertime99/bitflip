import argparse
import torch
import pandas as pd
import os
import utils
from preprocess import preprocess
from malconv import Malconv
from train import Custom_Dataset
from torch.utils.data import DataLoader, Dataset
import numpy as np

parser = argparse.ArgumentParser(description='Malconv-PyTorch Classifier')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--verbose', type=int, default=1)
parser.add_argument('--limit', type=float, default=0.)
parser.add_argument('--model_path', type=str, default='torch_saved/best_bs=256_lr=0.001_wd=0.001.pt')
# Test: Acc: 0.8086, Malware Acc: 0.8193, Benign Acc: 0.7980
parser.add_argument('--result_path', type=str, default='result.csv')
parser.add_argument('csv', type=str)
parser.add_argument('--save_dir', type=str, default='torch_saved')

def predict(model, dataloader, device):
    model.eval()
    
    correct, total = 0, 0
    malware_correct, malware_total = 0, 0
    benign_correct, benign_total = 0, 0
    all_predictions = [] 
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze()
            predictions = (outputs > 0.5).float()
            all_predictions.append(outputs.cpu())  
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
            malware_correct += (predictions[targets == 0] == 0).sum().item()
            malware_total += (targets == 0).sum().item()
            benign_correct += (predictions[targets == 1] == 1).sum().item()
            benign_total += (targets == 1).sum().item()
            
    test_acc = correct / total
    malware_acc = malware_correct / malware_total
    benign_acc = benign_correct / benign_total
    print(f'Test: Acc: {test_acc:.4f}, Malware Acc: {malware_acc:.4f}, Benign Acc: {benign_acc:.4f}')
    
    all_predictions = torch.cat(all_predictions, dim=0)
    return all_predictions

if __name__ == '__main__':
    args = parser.parse_args()
    
    # limit GPU memory
    if args.limit > 0:
        utils.limit_gpu_memory(args.limit)
    
    # load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Malconv()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # read data
    df = pd.read_csv(args.csv, header=None)
    fn_list = df[0].values
    
    # load data
    save_dir = args.save_dir
    x_test = torch.load(os.path.join(save_dir, 'x_test_set.pt'))
    y_test = torch.load(os.path.join(save_dir, 'y_test_set.pt'))
    test_dataset = Custom_Dataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    pred = predict(model, test_loader, device)
    df['predict score'] = pred
    df[0] = [os.path.basename(i) for i in fn_list]
    df.to_csv(args.result_path, header=None, index=False)
    print('Results written in', args.result_path)