import os
import time
import pickle
import argparse
import pandas as pd

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from utils import train_test_split

def preprocess(fn_list, max_len):
    '''
    Return processed data (ndarray) and original file length (list)
    '''
    import time 
    start_time = time.time()
    corpus = []
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"当前使用的设备: GPU ({torch.cuda.get_device_name(device)})")
    else:
        print("当前使用的设备: CPU")
        
    for index, fn in enumerate(fn_list):
        if (index + 1) % 1000 == 0:
            print(f'已处理 {index + 1} 个文件')
        
        if not os.path.isfile(fn):
            print(fn, 'not exist')
        else:
            #print(fn)
            with open(fn, 'rb') as f:
                corpus.append(f.read())

    len_list = [len(doc) for doc in corpus]
    
    np_corpus_list = []
    for doc in corpus:
        doc = doc[:max_len]
        np_doc = np.frombuffer(doc, dtype=np.uint8).astype(np.int32)
        pad_size = max_len - np_doc.size
        padded_doc = np.pad(np_doc, pad_width=(0, pad_size), mode='constant', constant_values=0).reshape((1,-1))
        np_corpus_list.append(padded_doc)
        
    corpus = np.concatenate(np_corpus_list, axis=0)
    
    print("Preprocess Finish")
    return corpus, len_list

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Malconv-PyTorch Preprocessor')
    parser.add_argument('--val_size', type=float, default=0.2, help="Validation split percentage")
    parser.add_argument('--max_len', type=int, default=200000) # 一个apk最多保留200k
    parser.add_argument('--save_dir', type=str, default='data')
    parser.add_argument('--csv', type=str, default='test_set1.csv')
    args = parser.parse_args()
    
    print("Preprocess data")
    df = pd.read_csv(args.csv, header=None)
    data, labels = df[0].values, df[1].values
    save_dir = args.save_dir
    
    benign_data = data[labels == 1]
    malware_data = data[labels == 0]
    
    print("Processing benign data...")
    x_benign, _ = preprocess(benign_data, args.max_len)  # 良性样本
    print("Processing malware data...")
    x_malware, _ = preprocess(malware_data, args.max_len)  # 恶意样本
    
    # 保存数据
    benign_path = os.path.join(save_dir, 'benign.pt')
    malware_path = os.path.join(save_dir, 'malware.pt')
    
    torch.save(x_benign, benign_path, pickle_protocol=4)
    torch.save(x_malware, malware_path, pickle_protocol=4)
    
    print(f'Benign data saved to: {benign_path}')
    print(f'Malware data saved to: {malware_path}')
    
    #x_test, _ = preprocess(data, args.max_len) # 测试集
    #x_test = torch.save(x_test, os.path.join(save_dir, 'x_test_set.pt'), pickle_protocol=4)
    #y_test = torch.save(labels, os.path.join(save_dir, 'y_test_set.pt'), pickle_protocol=4)    
    
    #x_train_name, x_test_name, y_train, y_test = train_test_split(data, labels, args.val_size)
    #print('Train on %d data, test on %d data' % (len(x_train_name), len(x_test_name)))
    
    # x_test, _ = preprocess(x_test_name, args.max_len)   # 验证集
    # x_test = torch.save(x_test, os.path.join(save_dir, 'x_test.pt'), pickle_protocol=4)
    # y_test = torch.save(y_test, os.path.join(save_dir, 'y_test.pt'), pickle_protocol=4)
    
    #x_train, _ = preprocess(x_train_name, args.max_len)
    #big_batch_size = 5000
    #for i in range(0,len(y_train),big_batch_size):
    #    x_batch = x_train[i:i+big_batch_size]
    #    y_batch = y_train[i:i+big_batch_size]
    #    x_batch = torch.save(x_batch, os.path.join(save_dir, f'x_train_{i//big_batch_size}.pt'), pickle_protocol=4)
    #    y_batch = torch.save(y_batch, os.path.join(save_dir, f'y_train_{i//big_batch_size}.pt'), pickle_protocol=4)
    #    print(f'Saved batch {i//big_batch_size}')
    
    # x_train = torch.save(x_train, os.path.join(save_dir, 'x_train.pt'), pickle_protocol=4)
    # y_train = torch.save(y_train, os.path.join(save_dir, 'y_train.pt'), pickle_protocol=4)

    exit()
    