from my_utils.model import load_model
from my_utils.load_data import train_model_data, train_model_data_ratio
from my_utils.utils import set_seed
from my_train import train

import torch
import torch.optim as optim

# ben_path = '/home/sample/lkc/torch_version/msdroid/DataNew/Benign/HOP_2/TPL_True/aux_datset.pt'
# mal_path = '/home/sample/lkc/torch_version/msdroid/DataNew/Drebin/HOP_2/TPL_True/aux_datset.pt'

ben_path = '/home/sample/lkc/torch_version/msdroid/DataNew/Benign/HOP_2/TPL_True/dataset.pt'
mal_path = '/home/sample/lkc/torch_version/msdroid/DataNew/Drebin/HOP_2/TPL_True/dataset.pt'


save_path = 'model/best.pt'
learning_rate = 0.001
def prepare_for_training():
    set_seed()
    gnn_model = load_model(layer_norm=False)
    train_data_list, test_data_list = train_model_data(ben_path, mal_path, train_size=4000, test_size=1000) # 4000 1000
    optimizer = optim.Adam(gnn_model.parameters(), lr=learning_rate)
    print('Start Training')
    train(gnn_model, train_data_list, test_data_list, apk_batch_size=64, epochs=100, val_epoch=1, optimizer=optimizer, save_path=save_path)
    
if __name__ == "__main__":
    import argparse
    prepare_for_training()

