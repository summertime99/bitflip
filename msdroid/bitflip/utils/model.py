#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@Time    :   2020/07/05 22:07:45
@Author  :   Yiling He
@Version :   1.0
@Contact :   heyilinge0@gmail.com
@License :   (C)Copyright 2020
@Desc    :   Define loss, build model, enable train and test. 
'''

# here put the import lib
from torch.nn import Sequential
import torch
import torch.nn.functional as F

import torch.nn as nn
import torch_geometric.nn as pyg_nn

import bitsandbytes as bnb

import numpy as np
import logging
model_logger = logging.getLogger()

# origin model unchange
class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv_func=None, global_pool=None, train_eps=False, layer_norm=False):
        super(GNNStack, self).__init__()
        self.convs = nn.ModuleList()
        self.conv_func = conv_func
        self.train_eps = train_eps
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.norm = nn.ModuleList()
        if layer_norm:
            self.norm.append(nn.LayerNorm(hidden_dim))
            self.norm.append(nn.LayerNorm(hidden_dim))
        else:
            self.norm.append(pyg_nn.BatchNorm(hidden_dim))
            self.norm.append(pyg_nn.BatchNorm(hidden_dim))

        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
        self.global_pool = global_pool

        # post-message-passing
        if self.global_pool == 'mix':
            self.post_mp = nn.Sequential(
                # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->??                # nn.Linear(hidden_dim*2, hidden_dim*2), nn.ReLU(inplace=True), # mix_relu
                nn.Linear(hidden_dim*2, hidden_dim), nn.Dropout(0.25), 
                nn.Linear(hidden_dim, output_dim))
        else:
            self.post_mp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25), 
                nn.Linear(hidden_dim, output_dim))

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # 训练使用的不是GATConv，是
        # (0): Linear(in_features=128, out_features=128, bias=True)
        # (1): ReLU()
        # (2): Linear(in_features=128, out_features=128, bias=True)
        # pyg_nn.GINConv的作用是根据边连接执行加法，然后对每一个节点的特征向量执行nn.Sequential()的操作
        if not self.conv_func:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                                nn.Linear(hidden_dim, hidden_dim)), train_eps=self.train_eps)
        elif self.conv_func == 'GATConv':
            return pyg_nn.GATConv(input_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index) # 图连接
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training) # training阶段apply，test阶段不使用
            if not i == self.num_layers - 1:
                x = self.norm[i](x)

        if not self.global_pool:
            x = pyg_nn.global_mean_pool(x, batch)
        elif self.global_pool == 'max':
            x = pyg_nn.global_max_pool(x, batch)
        elif self.global_pool == 'mix':
            x1 = pyg_nn.global_mean_pool(x, batch)
            x2 = pyg_nn.global_max_pool(x, batch)
            x = torch.cat((x1, x2), 1)

        emb = x # 最后一个api graph的embedding [hidden_dim*2]
        x = self.post_mp(x) # 一个分类的vector [2]
        out = F.log_softmax(x, dim=1)

        return emb, out, x
    
    def apk_loss(self, pred, label, position):
        loss = 0
        for i in range(len(position)-1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            apk_label = label[start:end]
            unilabel = set(apk_label.tolist())
            
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            if not unilabel: # Benign
                apk_loss = F.nll_loss(apk_pred, apk_label)  # log_softmax + nll_loss => cross_entropy
                # print('Benign Loss: %f' % apk_loss.item())
            else:
                scores = []
                for j in range(end-start):
                    scores.append(F.nll_loss(apk_pred[j:j+1], apk_label[j:j+1]))
                apk_loss = min(scores)
                # print('Malware Loss: %f' % apk_loss.item())
            
            loss += apk_loss
        return loss

    def apk_hard_loss(self, pred, label, position, weights=True):
        loss = 0
        for i in range(len(position)-1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            apk_label = label[start:end]
            unilabel = set(apk_label.tolist())
            
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            if not unilabel: # Benign
                apk_loss = F.nll_loss(apk_pred, apk_label)
                # print('Benign Loss: %f' % apk_loss.item())
            else:
                scores = []
                all_scores = []
                for j in range(end-start):
                    single_pred = apk_pred[j:j+1]
                    single_loss = F.nll_loss(apk_pred[j:j+1], apk_label[j:j+1])
                    all_scores.append(single_loss)
                    if single_pred.argmax(dim=1):
                        scores.append(single_loss)
                sclen = len(scores)
                if sclen:
                    if weights:
                        w = np.linspace(0, 1, num=sclen+1)
                        w = (w / sum(w))[1:]
                        scores.sort(reverse=True) # descending order(larger loss, smaller weight??                        apk_loss = 0
                        for i in range(len(w)):
                            apk_loss += scores[i]*w[i]  
                    else:
                        apk_loss = sum(scores) / len(scores)
                else:
                    apk_loss =  min(all_scores)
                # print('Malware Loss: %f' % apk_loss.item())
            
            loss += apk_loss
        return loss

# 所有都做量化    
class GNNStack_INT8(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv_func=None, global_pool=None, train_eps=False, layer_norm=False):
        super(GNNStack_INT8, self).__init__()
        self.convs = nn.ModuleList()
        self.conv_func = conv_func
        self.train_eps = train_eps
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.norm = nn.ModuleList()
        if layer_norm:
            self.norm.append(nn.LayerNorm(hidden_dim).to(torch.float16))
            self.norm.append(nn.LayerNorm(hidden_dim).to(torch.float16))
        else:
            self.norm.append(pyg_nn.BatchNorm(hidden_dim))
            self.norm.append(pyg_nn.BatchNorm(hidden_dim))

        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
        self.global_pool = global_pool

        # post-message-passing
        if self.global_pool == 'mix':
            self.post_mp = nn.Sequential(
                # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->??                # nn.Linear(hidden_dim*2, hidden_dim*2), nn.ReLU(inplace=True), # mix_relu
                bnb.nn.Linear8bitLt(hidden_dim*2, hidden_dim, has_fp16_weights=False), nn.Dropout(0.25), 
                bnb.nn.Linear8bitLt(hidden_dim, output_dim, has_fp16_weights=False))
        else:
            self.post_mp = nn.Sequential(
                bnb.nn.Linear8bitLt(hidden_dim, hidden_dim, has_fp16_weights=False), nn.Dropout(0.25), 
                bnb.nn.Linear8bitLt(hidden_dim, output_dim, has_fp16_weights=False))

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # 训练使用的不是GATConv，是
        # (0): Linear(in_features=128, out_features=128, bias=True)
        # (1): ReLU()
        # (2): Linear(in_features=128, out_features=128, bias=True)
        # pyg_nn.GINConv的作用是根据边连接执行加法，然后对每一个节点的特征向量执行nn.Sequential()的操作
        if not self.conv_func:
            return pyg_nn.GINConv(nn.Sequential(bnb.nn.Linear8bitLt(input_dim, hidden_dim, has_fp16_weights=False), nn.ReLU(),
                                                bnb.nn.Linear8bitLt(hidden_dim, hidden_dim, has_fp16_weights=False)), train_eps=self.train_eps)
        elif self.conv_func == 'GATConv':
            return pyg_nn.GATConv(input_dim, hidden_dim)

    def forward(self, data):
        # 一个batch输入，假定整合了a个子图，输出就是 emb:[a, hidden_dim*2], pred:[a,2]
        # 这里不对输入处理，因为pyg_nn 处理后会变成float32
        # 留到my_linear内部处理
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if data.num_node_features == 0:
            # print('called num_node_features')
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index) # 图连接

            x = F.relu(x)

            x = F.dropout(x, p=self.dropout, training=self.training) # training阶段apply，test阶段不使用
            x = x.to(torch.float16)
            # print('x before norm', x.dtype)
            # print('norm func', self.norm[i])
            if not i == self.num_layers - 1:
                x = self.norm[i](x)

        if not self.global_pool:
            x = pyg_nn.global_mean_pool(x, batch)
        elif self.global_pool == 'max':
            x = pyg_nn.global_max_pool(x, batch)
        elif self.global_pool == 'mix':
            x1 = pyg_nn.global_mean_pool(x, batch)
            x2 = pyg_nn.global_max_pool(x, batch)
            x = torch.cat((x1, x2), 1)

        emb = x # 最后一个api graph的embedding [hidden_dim*2]
        x = self.post_mp(x) # 一个分类的vector [2]
        out = F.log_softmax(x, dim=1)

        return emb, out, x
    
    def apk_loss(self, pred, label, position):
        loss = 0
        for i in range(len(position)-1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            apk_label = label[start:end]
            unilabel = set(apk_label.tolist())
            
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            if not unilabel: # Benign
                apk_loss = F.nll_loss(apk_pred, apk_label)  # log_softmax + nll_loss => cross_entropy
                # print('Benign Loss: %f' % apk_loss.item())
            else:
                scores = []
                for j in range(end-start):
                    scores.append(F.nll_loss(apk_pred[j:j+1], apk_label[j:j+1]))
                apk_loss = min(scores)
                # print('Malware Loss: %f' % apk_loss.item())
            
            loss += apk_loss
        return loss

    def apk_hard_loss(self, pred, label, position, weights=True):
        loss = 0
        for i in range(len(position)-1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            apk_label = label[start:end]
            unilabel = set(apk_label.tolist())
            
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            if not unilabel: # Benign
                apk_loss = F.nll_loss(apk_pred, apk_label)
                # print('Benign Loss: %f' % apk_loss.item())
            else:
                scores = []
                all_scores = []
                for j in range(end-start):
                    single_pred = apk_pred[j:j+1]
                    single_loss = F.nll_loss(apk_pred[j:j+1], apk_label[j:j+1])
                    all_scores.append(single_loss)
                    if single_pred.argmax(dim=1):
                        scores.append(single_loss)
                sclen = len(scores)
                if sclen:
                    if weights:
                        w = np.linspace(0, 1, num=sclen+1)
                        w = (w / sum(w))[1:]
                        scores.sort(reverse=True) # descending order(larger loss, smaller weight??                        apk_loss = 0
                        for i in range(len(w)):
                            apk_loss += scores[i]*w[i]  
                    else:
                        apk_loss = sum(scores) / len(scores)
                else:
                    apk_loss =  min(all_scores)
                # print('Malware Loss: %f' % apk_loss.item())
            
            loss += apk_loss
        return loss
# 只对GNN的输出之后的两层MLP做量化
class GNNStack_INT8_2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, conv_func=None, global_pool=None, train_eps=False, layer_norm=False):
        super(GNNStack_INT8_2, self).__init__()
        self.convs = nn.ModuleList()
        self.conv_func = conv_func
        self.train_eps = train_eps
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.norm = nn.ModuleList()
        if layer_norm:
            self.norm.append(nn.LayerNorm(hidden_dim))
            self.norm.append(nn.LayerNorm(hidden_dim))
        else:
            self.norm.append(pyg_nn.BatchNorm(hidden_dim))
            self.norm.append(pyg_nn.BatchNorm(hidden_dim))

        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))
        self.global_pool = global_pool

        # post-message-passing
        if self.global_pool == 'mix':
            self.post_mp = nn.Sequential(
                # -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->??                # nn.Linear(hidden_dim*2, hidden_dim*2), nn.ReLU(inplace=True), # mix_relu
                bnb.nn.Linear8bitLt(hidden_dim*2, hidden_dim, has_fp16_weights=False), nn.Dropout(0.25), 
                bnb.nn.Linear8bitLt(hidden_dim, output_dim, has_fp16_weights=False))
        else:
            self.post_mp = nn.Sequential(
                bnb.nn.Linear8bitLt(hidden_dim, hidden_dim, has_fp16_weights=False), nn.Dropout(0.25), 
                bnb.nn.Linear8bitLt(hidden_dim, output_dim, has_fp16_weights=False))

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # 训练使用的不是GATConv，是
        # (0): Linear(in_features=128, out_features=128, bias=True)
        # (1): ReLU()
        # (2): Linear(in_features=128, out_features=128, bias=True)
        # pyg_nn.GINConv的作用是根据边连接执行加法，然后对每一个节点的特征向量执行nn.Sequential()的操作
        if not self.conv_func:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                                nn.Linear(hidden_dim, hidden_dim)), train_eps=self.train_eps)
        elif self.conv_func == 'GATConv':
            return pyg_nn.GATConv(input_dim, hidden_dim)

    def forward(self, data):
        # 一个batch输入，假定整合了a个子图，输出就是 emb:[a, hidden_dim*2], pred:[a,2]
        # 这里不对输入处理，因为pyg_nn 处理后会变成float32
        # 留到my_linear内部处理
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        if data.num_node_features == 0:
            # print('called num_node_features')
            x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index) # 图连接

            x = F.relu(x)

            x = F.dropout(x, p=self.dropout, training=self.training) # training阶段apply，test阶段不使用
            if not i == self.num_layers - 1:
                x = self.norm[i](x)

        if not self.global_pool:
            x = pyg_nn.global_mean_pool(x, batch)
        elif self.global_pool == 'max':
            x = pyg_nn.global_max_pool(x, batch)
        elif self.global_pool == 'mix':
            x1 = pyg_nn.global_mean_pool(x, batch)
            x2 = pyg_nn.global_max_pool(x, batch)
            x = torch.cat((x1, x2), 1)
        x = x.to(torch.float16)
        emb = x # 最后一个api graph的embedding [hidden_dim*2]
        x = self.post_mp(x) # 一个分类的vector [2]
        out = F.log_softmax(x, dim=1)

        return emb, out, x
    
    def apk_loss(self, pred, label, position):
        loss = 0
        for i in range(len(position)-1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            apk_label = label[start:end]
            unilabel = set(apk_label.tolist())
            
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            if not unilabel: # Benign
                apk_loss = F.nll_loss(apk_pred, apk_label)  # log_softmax + nll_loss => cross_entropy
                # print('Benign Loss: %f' % apk_loss.item())
            else:
                scores = []
                for j in range(end-start):
                    scores.append(F.nll_loss(apk_pred[j:j+1], apk_label[j:j+1]))
                apk_loss = min(scores)
                # print('Malware Loss: %f' % apk_loss.item())
            
            loss += apk_loss
        return loss

    def apk_hard_loss(self, pred, label, position, weights=True):
        loss = 0
        for i in range(len(position)-1):
            start, end = position[i:i+2]
            apk_pred = pred[start:end]
            apk_label = label[start:end]
            unilabel = set(apk_label.tolist())
            
            assert len(unilabel)==1
            unilabel = list(unilabel)[0]
            if not unilabel: # Benign
                apk_loss = F.nll_loss(apk_pred, apk_label)
                # print('Benign Loss: %f' % apk_loss.item())
            else:
                scores = []
                all_scores = []
                for j in range(end-start):
                    single_pred = apk_pred[j:j+1]
                    single_loss = F.nll_loss(apk_pred[j:j+1], apk_label[j:j+1])
                    all_scores.append(single_loss)
                    if single_pred.argmax(dim=1):
                        scores.append(single_loss)
                sclen = len(scores)
                if sclen:
                    if weights:
                        w = np.linspace(0, 1, num=sclen+1)
                        w = (w / sum(w))[1:]
                        scores.sort(reverse=True) # descending order(larger loss, smaller weight??                        apk_loss = 0
                        for i in range(len(w)):
                            apk_loss += scores[i]*w[i]  
                    else:
                        apk_loss = sum(scores) / len(scores)
                else:
                    apk_loss =  min(all_scores)
                # print('Malware Loss: %f' % apk_loss.item())
            
            loss += apk_loss
        return loss

