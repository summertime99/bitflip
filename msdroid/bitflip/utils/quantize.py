import torch
from torch.utils.data import DataLoader
import numpy as np

from model import GNNStack
# from utils import replace_linear_with_8bit, load_model

# dataset_list: dataset的路径列表
def load_dataset(dataset_list)->DataLoader:
    pass


if __name__ == "__main__":
    gnn_path = '/home/sample/lkc/MsDroid/src/training/Experiments/20250130-134544/models/last_epoch_200'
    # gnn_model = load_model(gnn_path)
    # replace_linear_with_8bit(gnn_model)

    # print(gnn_model)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # gnn_model.to(device)
    
    exit()
    gnn_model.eval()

    # load data: 数据都是0,1
    benign_data = torch.tensor(np.load(benign_path).astype(np.float32)).to(device)
    malware_data = torch.tensor(np.load(malware_path).astype(np.float32)).to(device)
    sample_num = min(benign_data.shape[0], malware_data.shape[0])

    benign_data = benign_data[:sample_num, :].to(device)
    malware_data = malware_data[:sample_num, :].to(device)
    print('data_shape:', benign_data.shape, malware_data.shape)

    # mu, sigma, output
    mlp_benign_output, vae_benign_output, _, _ = robust_model(benign_data)
    mlp_malware_output, vae_malware_output, _, _ = robust_model(malware_data)

    mlp_benign_predicted = torch.argmax(mlp_benign_output, dim=1)
    model_predicted = mlp_benign_predicted # * loss1_predicted
    benign_correct_count = torch.eq(model_predicted, torch.zeros(model_predicted.shape).to(torch.int).to(device)).sum().item()

    mlp_malware_predicted = torch.argmax(mlp_malware_output, dim=1)
    model_predicted = mlp_malware_predicted # * loss1_predicted
    malware_correct_count = torch.eq(model_predicted, torch.ones(model_predicted.shape).to(torch.int).to(device)).sum().item()

    print("mal/ben num:{}, mal correct:{}, ben_correct:{}".format(sample_num, malware_correct_count, benign_correct_count))
    print("acc:{}".format((malware_correct_count + benign_correct_count) / (2 * sample_num)))

