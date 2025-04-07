import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def limit_gpu_memory(per):
    torch.cuda.set_per_process_memory_fraction(per)

def train_test_split(data, label, val_ratio=0.1):
    np.random.seed(666)
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    print('idx sample',idx[0:20])
    
    split = int(len(data) * val_ratio)
    x_train, x_test = data[idx[split:]], data[idx[:split]]
    y_train, y_test = label[idx[split:]], label[idx[:split]]
    print("train_test_split return")
    return x_train, x_test, y_train, y_test

class logger:
    def __init__(self):
        self.fn = []
        self.len = []
        self.pad_len = []
        self.loss = []
        self.pred = []
        self.org = []
    
    def write(self, fn, org_score, file_len, pad_len, loss, pred):
        self.fn.append(fn.split('/')[-1])
        self.org.append(org_score)
        self.len.append(file_len)
        self.pad_len.append(pad_len)
        self.loss.append(loss)
        self.pred.append(pred)
        
        print('\nFILE:', fn)
        if pad_len > 0:
            print('\tfile length:', file_len)
            print('\tpad length:', pad_len)
            print('\tloss:', loss)
            print('\tscore:', pred)
        else:
            print('\tfile length:', file_len, ', Exceed max length ! Ignored !')
        print('\toriginal score:', org_score)
    
    def save(self, path):
        d = {'filename': self.fn, 
             'original score': self.org, 
             'file length': self.len,
             'pad length': self.pad_len, 
             'loss': self.loss, 
             'predict score': self.pred}
        df = pd.DataFrame(data=d)
        df.to_csv(path, index=False, columns=['filename', 'original score', 
                                              'file length', 'pad length', 
                                              'loss', 'predict score'])
        print('\nLog saved to "%s"\n' % path)
