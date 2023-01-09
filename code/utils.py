# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:33:36 2020

@author: zhangxf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# my optimizer
class supSGD():
    def __init__(self, params, lr=0.01, weight_decay=0.):
        self.lr = lr
        self.params=params
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if self.weight_decay != 0:
                d_p.add_(self.weight_decay, p.data)
            p.data.add_(-self.lr, d_p)

    def update_lr(self,weight):
        self.lr *= weight
        return self.lr

# pooling method
def pooling(inputs, pooling_type='avg'):
    if pooling_type == 'avg':
        return torch.mean(inputs,dim=1)
    elif pooling_type == 'max':
        return torch.max(inputs, 1)[0]
    elif pooling_type == 'min':
        return torch.min(inputs, 1)[0]
    
def act_f(inputs,act = "null"):
    if act == "sigmoid":
        return torch.sigmoid(inputs)
    elif act == "tanh":
        return torch.tanh(inputs)
    elif act == "relu":
        return torch.relu(inputs)
    elif act == "nrelu":
        #return torch.neg(torch.relu(torch.neg(inputs)))
        return torch.min(inputs,torch.tensor(0).float().to(inputs.device))
    elif act == "null":
        return inputs

def del_edge(anc_all, anc, S_adjs, T_adjs, ht1, ht2):
    ind = [[],[]]
    val = []
    for i in anc_all:
        tmp = ht1[i].keys() & ht2[i].keys() & anc
        for j in tmp:
            ind[0].append(i)
            ind[1].append(j)
            val.append(1)
            ind[0].append(j)
            ind[1].append(i)
            val.append(1)

    ind = torch.LongTensor(ind)
    val = torch.FloatTensor(val)
    k1 = torch.sparse.FloatTensor(ind, val, S_adjs.size()).cuda()
    k2 = torch.sparse.FloatTensor(ind, val, T_adjs.size()).cuda()
    print("remove edge: %d/%d of s | %d/%d of t"%(torch.sum(val), torch.sum(S_adjs._values()), torch.sum(val), torch.sum(T_adjs._values())))
    return S_adjs - k1, T_adjs - k2

def add_edge(anc_all, anc, S_adjs, T_adjs, ht1, ht2):
    s_ind = [[],[]]
    s_val = []
    t_ind = [[],[]]
    t_val = []
    for i in anc_all:
        t_tmp = (ht1[i].keys() & anc) - ht2[i].keys()
        for j in t_tmp:
            t_ind[0].append(i)
            t_ind[1].append(j)
            t_val.append(1)
            t_ind[0].append(j)
            t_ind[1].append(i)
            t_val.append(1)
        s_tmp = (ht2[i].keys() & anc) - ht1[i].keys()
        for j in s_tmp:
            s_ind[0].append(i)
            s_ind[1].append(j)
            s_val.append(1)
            s_ind[0].append(j)
            s_ind[1].append(i)
            s_val.append(1)

    s_ind = torch.LongTensor(s_ind)
    s_val = torch.FloatTensor(s_val)
    s = torch.sparse.FloatTensor(s_ind, s_val, S_adjs.size()).cuda()
    t_ind = torch.LongTensor(t_ind)
    t_val = torch.FloatTensor(t_val)
    t = torch.sparse.FloatTensor(t_ind, t_val, T_adjs.size()).cuda()
    print("remove edge: %d/%d of s | %d/%d of t"%(torch.sum(s_val), torch.sum(S_adjs._values()), torch.sum(t_val), torch.sum(T_adjs._values())))
    return S_adjs + s, T_adjs + t


def load_SRPemb(path):
	f1 = open(path)
	vectors = []
	for i, line in enumerate(f1):
		id, word, vect = line.rstrip().split('\t', 2)
		vect = np.fromstring(vect, sep=' ')
		vectors.append(vect)
	embeddings = np.vstack(vectors)
	return embeddings