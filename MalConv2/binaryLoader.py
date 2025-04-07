import os
from collections import deque

import random
import numpy as np

from tqdm import tqdm_notebook as tqdm

import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import torch.optim as optim

from torch.utils import data

from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd


class BinaryDataset(data.Dataset):
    """
    Loader for binary files. 
    
    If you use the sort_by_file_size option, the dataset will store files from smallest to largest. This is meant to used with RandomChunkSampler to sammple batches of similarly sized files to maximize performance. 
    
    TODO: Auto un-gzip files if they have g-zip compression 
    """
    def __init__(self, good_csv, bad_csv, sort_by_size=False, max_len=4000000):
        
        #Tuple (file_path, label, file_size)
        self.all_files = []
        self.max_len = max_len
        
        good_df = pd.read_csv(good_csv, header=None)
        for file in good_df[0]:
            if os.path.exists(file):
                self.all_files.append((file, 0, os.path.getsize(file)))

        bad_df = pd.read_csv(bad_csv, header=None)
        for file in bad_df[0]:
            if os.path.exists(file):
                self.all_files.append((file, 1, os.path.getsize(file)))
                
        if sort_by_size:
            self.all_files.sort(key=lambda filename: filename[2])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        
        to_load, y, _ = self.all_files[index]
        
        try:
            with gzip.open(to_load, 'rb') as f:
                x = f.read(self.max_len)
                #Need to use frombuffer b/c its a byte array, otherwise np.asarray will get wonked on trying to convert to ints
                #So decode as uint8 (1 byte per value), and then convert
                x = np.frombuffer(x, dtype=np.uint8).astype(np.int16)+1 #index 0 will be special padding index
        except OSError:
            #OK, you are not a gziped file. Just read in raw bytes from disk. 
            with open(to_load, 'rb') as f:
                x = f.read(self.max_len)
                #Need to use frombuffer b/c its a byte array, otherwise np.asarray will get wonked on trying to convert to ints
                #So decode as uint8 (1 byte per value), and then convert
                x = np.frombuffer(x, dtype=np.uint8).astype(np.int16)+1 #index 0 will be special padding index
            
        #x = np.pad(x, self.max_len-x.shape[0], 'constant')    
        x = torch.tensor(x)

        return x, torch.tensor([y])
    
class RandomChunkSampler(torch.utils.data.sampler.Sampler):
    """
    Samples random "chunks" of a dataset, so that items within a chunk are always loaded together. Useful to keep chunks in similar size groups to reduce runtime. 
    """
    def __init__(self, data_source, batch_size):
        """
        data_source: the souce pytorch dataset object
        batch_size: the size of the chunks to keep together. Should generally be set to the desired batch size during training to minimize runtime. 
        """
        self.data_source = data_source
        self.batch_size = batch_size
        
    def __iter__(self):
        n = len(self.data_source)
        
        data = [x for x in range(n)]

        # Create blocks
        blocks = [data[i:i+self.batch_size] for i in range(0,len(data),self.batch_size)]
        # shuffle the blocks
        random.shuffle(blocks)
        # concatenate the shuffled blocks
        data[:] = [b for bs in blocks for b in bs]
        
        return iter(data)
        
    def __len__(self):
        return len(self.data_source)
    
#We want to hadnel true variable length
#Data loader needs equal length. So use special function to padd all the data in a single batch to be of equal length
#to the longest item in the batch
def pad_collate_func(batch):
    """
    This should be used as the collate_fn=pad_collate_func for a pytorch DataLoader object in order to pad out files in a batch to the length of the longest item in the batch. 
    """
    vecs = [x[0] for x in batch]
    labels = [x[1] for x in batch]
    
    x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True)
    #stack will give us (B, 1), so index [:,0] to get to just (B)
    y = torch.stack(labels)[:,0]
    
    return x, y