# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:33:16 2020

@author: zhangxf
"""

import random
import argparse
import time
import gc
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from code.utils import pooling, act_f


class Linear_trans(nn.Module):
    def __init__(self, opt, pre_emb):
        super().__init__()
        self.opt = opt
        # embedding layer
        if opt["pretrained_embedding"] == "T":
            self.Eemb = nn.Embedding(self.opt['node_num_1'] + self.opt['node_num_2'], self.opt['input_dim'], max_norm=1, _weight=pre_emb)
            #self.Eemb = nn.Embedding.from_pretrained(pre_emb,freeze=False,max_norm=1)
        else:
            self.Eemb = nn.Embedding(self.opt['node_num_1'] + self.opt['node_num_2'], self.opt['input_dim'], max_norm=1)
            #TKDE change
            nn.init.xavier_uniform_(self.Eemb.weight.data)
        #nn.init.xavier_uniform_(self.Eemb.weight.data)
        self.L1 = nn.Linear(self.opt['input_dim'], self.opt['output_dim'],bias=False)
        if self.opt["init"] == "x_u":
            nn.init.xavier_uniform_(self.L1.weight,self.opt["gain"])
        elif self.opt["init"] == "x_n":
            nn.init.xavier_normal_(self.L1.weight,self.opt["gain"])
        
        for p in self.L1.parameters():
            p.requires_grad=False
    
    def set_pretrained_embedding(self, emb):
        self.Eemb = nn.Embedding(emb)

    def forward(self, S_in, T_in, anc, s_weight, t_weight):
        S_inputs = self.Eemb(S_in)
        T_inputs = self.Eemb(T_in)
        if self.opt["augmentation"] == "T":
            S_tmp = self.L1(S_inputs)
            T_tmp = self.L1(T_inputs)
        else:
            S_tmp = S_inputs
            T_tmp = T_inputs
        S_out = S_tmp
        T_out = T_tmp
        S_out *= s_weight.unsqueeze(1)
        T_out *= t_weight.unsqueeze(1)
        
        return S_out, T_out

class RankGCN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        self.bias1 = nn.Parameter(torch.rand(self.opt['node_num_1'], int(self.opt["output_dim"]/self.opt["sub_model"])))
        self.bias1.requires_grad = True
        self.bias2 = nn.Parameter(torch.rand(self.opt['node_num_2'], int(self.opt["output_dim"]/self.opt["sub_model"])))
        self.bias2.requires_grad = True

        nn.init.normal_(self.bias1,0,1)
        nn.init.normal_(self.bias2,0,1)
        self.pooling_weight = nn.Parameter(torch.randn(self.opt["pw"],self.opt["div"]).t())
        self.pooling_weight.requires_grad = False
        self.pooling_weight_last = nn.Parameter(torch.randn(self.opt["lpw"],self.opt["ldiv"]).t())
        self.pooling_weight.requires_grad = False
        self.pooling_weight1 = nn.Parameter(torch.randn(self.opt["pw"],self.opt["div"]).t())
        self.pooling_weight1.requires_grad = False
        self.pooling_weight1_last = nn.Parameter(torch.randn(self.opt["lpw"],self.opt["ldiv"]).t())
        self.pooling_weight1.requires_grad = False
        
    
    def forward(self, S_ad, T_ad, S_eye, T_eye, S_ou, T_ou):
        Ama_loss = []
        Ami_loss = []
        Ame_loss = []
        Amr_loss = []
        Amrl_loss = []
        Amc_loss = []

        S_out = S_ou
        T_out = T_ou
        S_adjs = S_ad
        T_adjs = T_ad
        for i in range(self.opt["gcn_num"]):
            devi = "cuda:0"
            S_out = S_adjs.mm(S_out.squeeze(0)).unsqueeze(0).to(devi)
            T_out = T_adjs.mm(T_out.squeeze(0)).unsqueeze(0).to(devi)
            
            if self.opt["bias_flag"] == "T":
                S_tmp = torch.sub(S_out,self.bias1.to(devi))
                T_tmp = torch.sub(T_out,self.bias2.to(devi))
            else:
                S_tmp = S_out
                T_tmp = T_out
            
            if self.opt["noise_alpha"] >= 0:
                #print(S_tmp.size())
                S_tmp += make_noise(self.opt, S_adjs,devi)
                T_tmp += make_noise(self.opt, T_adjs,devi)
            
            if self.opt["ablation"] == "null":
                if self.opt["relu_flag"] == "T":
                    tma, tmi, tme = act_pooling(S_tmp, T_tmp)
                else:
                    tma, tmi, tme = mix_pooling(S_tmp, T_tmp)
            elif self.opt["ablation"] == "max":
                _, tmi, tme = act_pooling(S_tmp, T_tmp)
                tma = torch.tensor(0)
            elif self.opt["ablation"] == "min":
                tma, _, tme = act_pooling(S_tmp, T_tmp)
                tmi = torch.tensor(0)
            elif self.opt["ablation"] == "avg":
                tma, tmi, _ = act_pooling(S_tmp, T_tmp)
                tme = torch.tensor(0)
            elif self.opt["ablation"] == "all":
                tma = torch.tensor(0)
                tmi = torch.tensor(0)
                tme = torch.tensor(0)
            
            if self.opt["topk"] == 0:
                ra1 = torch.tensor(0)
                ra2 = torch.tensor(0)
            elif i == 0:
                ra1 = rank_mat_pooling(S_tmp, T_tmp, int(self.opt["topk"]/self.opt["div"]), self.opt["div"], self.pooling_weight.to(devi), largest=True)
                ra2 = rank_mat_pooling(S_tmp, T_tmp, int(self.opt["lastk"]/self.opt["ldiv"]), self.opt["ldiv"], self.pooling_weight_last.to(devi), largest=False)
            else :
                ra1 = rank_mat_pooling(S_tmp, T_tmp, int(self.opt["topk"]/self.opt["div"]), self.opt["div"], self.pooling_weight1.to(devi), largest=True)
                ra2 = rank_mat_pooling(S_tmp, T_tmp, int(self.opt["lastk"]/self.opt["ldiv"]), self.opt["ldiv"], self.pooling_weight1_last.to(devi), largest=False)
            
            
            Ama_loss.append(tma.to("cuda:0"))
            Ami_loss.append(tmi.to("cuda:0"))
            Ame_loss.append(tme.to("cuda:0"))
            Amr_loss.append(ra1.to("cuda:0"))
            Amrl_loss.append(ra2.to("cuda:0"))
        
        return Ama_loss, Ami_loss, Ame_loss, Amr_loss, Amrl_loss


def make_noise(op, Adj, dev):
    '''
    input : adj
    opt: noise_alpha
    '''
    siz = Adj.size()[0]
    if op["noise_type"] == "vector":
        noise = torch.randn(siz,op["output_dim"]).to(dev)
        noise = torch.pow(Adj.to(dev),op["noise_alpha"]).t().mm(noise)
    elif op["noise_type"] == "dif_scalar":
        noise = torch.sparse.FloatTensor(Adj._indices(), torch.randn(len(Adj._values())).to(dev), Adj.size()).to_dense()
        noise = torch.mean(torch.pow(Adj.to(dev),op["noise_alpha"]).t().mm(noise),dim=1).unsqueeze(1)
    elif op["noise_type"] == "same_scalar":
        noise = torch.randn(siz,1).to(dev)
        noise = torch.pow(Adj.to(dev),op["noise_alpha"]).t().mm(noise)
    return noise


def mix_pooling(s_inputs,t_inputs):
    S_pool = pooling(s_inputs, "max")
    T_pool = pooling(t_inputs, "max")
    tmp = torch.sub(S_pool,T_pool).abs()
    Pma_loss = torch.sum(torch.pow(tmp,2))
    
    S_pool = pooling(s_inputs, "min")
    T_pool = pooling(t_inputs, "min")
    tmp = torch.sub(S_pool,T_pool).abs()
    Pmi_loss = torch.sum(torch.pow(tmp,2))
    
    S_pool = pooling(s_inputs, "avg")
    T_pool = pooling(t_inputs, "avg")
    tmp = torch.sub(S_pool,T_pool).abs()
    Pav_loss = torch.sum(torch.pow(tmp,2))
    
    return Pma_loss , Pmi_loss , Pav_loss

def rank_pooling(s_inputs, t_inputs, k):
    s_i = torch.topk(torch.relu(s_inputs), k, dim=1, sorted = True)[0].squeeze(0)
    #size:k*20000,
    t_i = torch.topk(torch.relu(t_inputs), k, dim=1, sorted = True)[0].squeeze(0)
    tmp = torch.sub(torch.mean(s_i, dim=0), torch.mean(t_i, dim=0)).abs()
    return torch.sum(torch.pow(tmp,2))

def rank_weighted_pooling(s_inputs, t_inputs, k, div = 5, h_a = 0.5):
    s_i = torch.topk(torch.relu(s_inputs), k*div, dim=1, sorted = True)[0].squeeze(0)
    #size:k*20000,
    t_i = torch.topk(torch.relu(t_inputs), k*div, dim=1, sorted = True)[0].squeeze(0)
    tmp = 0
    for i in range(div):
        tmp += h_a*((1-h_a)**i)*torch.sub(torch.mean(s_i[i*k:(i+1)*k], dim=0), torch.mean(t_i[i*k:(i+1)*k], dim=0))
    return torch.sum(torch.pow(tmp,2))

def rank_rec_pooling(s_inputs, t_inputs, k, div = 5):
    s_i = torch.topk(torch.relu(s_inputs), k*div, dim=1, sorted = True)[0].squeeze(0)
    #size:k*20000,
    t_i = torch.topk(torch.relu(t_inputs), k*div, dim=1, sorted = True)[0].squeeze(0)
    tmp = 0
    for i in range(div):
        tmp += (1/(i+1))*torch.sub(torch.mean(s_i[i*k:(i+1)*k], dim=0), torch.mean(t_i[i*k:(i+1)*k], dim=0))
    return torch.sum(torch.pow(tmp,2))

def rank_mix_pooling(s_inputs, t_inputs, k, div = 5, h_a = 0.5):
    s_i = torch.topk(torch.relu(s_inputs), k*div, dim=1, sorted = True)[0].squeeze(0)
    #size:k*20000,
    t_i = torch.topk(torch.relu(t_inputs), k*div, dim=1, sorted = True)[0].squeeze(0)
    ml = torch.sum(torch.pow(torch.sub(torch.mean(s_i, dim=0), torch.mean(t_i, dim=0)).abs(),2))
    tmp = 0
    for i in range(div):
        tmp += h_a*((1-h_a)**i)*torch.sub(torch.mean(s_i[i*k:(i+1)*k], dim=0), torch.mean(t_i[i*k:(i+1)*k], dim=0))
    ml += torch.sum(torch.pow(tmp,2))
    del tmp
    tmp = 0
    for i in range(div):
        tmp += (1/(i+1))*torch.sub(torch.mean(s_i[i*k:(i+1)*k], dim=0), torch.mean(t_i[i*k:(i+1)*k], dim=0))
    ml += torch.sum(torch.pow(tmp,2))
    del tmp
    return ml

def rank_mat_pooling(s_inputs, t_inputs, k, div, p_w, largest=True):
    #s_i = torch.topk(torch.relu(s_inputs), k*div, dim=1, sorted = True)[0].squeeze(0)
    #size:k*20000,
    #t_i = torch.topk(torch.relu(t_inputs), k*div, dim=1, sorted = True)[0].squeeze(0)
    dev = s_inputs.device
    if largest:
        st = torch.sub(torch.topk(torch.relu(s_inputs), k*div, dim=1, sorted = True)[0].squeeze(0),torch.topk(torch.relu(t_inputs), k*div, dim=1, sorted = True)[0].squeeze(0))
    else:
        st = torch.sub(torch.topk(torch.min(s_inputs,torch.tensor(0).float().to(dev)), k*div, dim=1, largest=False, sorted = True)[0].squeeze(0),torch.topk(torch.min(t_inputs,torch.tensor(0).float().to(dev)), k*div, dim=1, largest=False, sorted = True)[0].squeeze(0))
    tmp = 0
    for i in range(div):
        tmp += torch.matmul(p_w[i].unsqueeze(1),torch.mean(st[i*k:(i+1)*k], dim=0).unsqueeze(0))
    return torch.sum(torch.sum(torch.pow(tmp,2),dim=1),dim=0)

def cov1_pooling(s_inputs,t_inputs):
    S_cov = torch.sub(s_inputs,pooling(s_inputs, "avg")).squeeze(0)
    T_cov = torch.sub(t_inputs,pooling(t_inputs, "avg")).squeeze(0)
    S_cov = torch.div(torch.mm(S_cov.t(),S_cov),s_inputs.size()[1])
    T_cov = torch.div(torch.mm(T_cov.t(),T_cov),t_inputs.size()[1])
    #S_cov = torch.mm(S_cov.t(),S_cov)
    #T_cov = torch.mm(T_cov.t(),T_cov)
    #stn = torch.norm(torch.sub(S_cov,T_cov),p=2)
    stn = torch.sum(torch.pow(torch.sub(S_cov,T_cov),2))
    del S_cov,T_cov
    return stn

def act_pooling(s_inputs,t_inputs):
    dev = s_inputs.device

    S_pool = pooling(torch.relu(s_inputs), "max")
    T_pool = pooling(torch.relu(t_inputs), "max")
    tmp = torch.sub(S_pool,T_pool).abs()
    Pma_loss = torch.sum(torch.pow(tmp,2))
    
    S_pool = pooling(torch.min(s_inputs,torch.tensor(0).float().to(dev)), "min")
    T_pool = pooling(torch.min(t_inputs,torch.tensor(0).float().to(dev)), "min")
    tmp = torch.sub(S_pool,T_pool).abs()
    Pmi_loss = torch.sum(torch.pow(tmp,2))
    
    S_pool = pooling(s_inputs, "avg")
    T_pool = pooling(t_inputs, "avg")
    tmp = torch.sub(S_pool,T_pool).abs()
    Pav_loss = torch.sum(torch.pow(tmp,2))
    
    del S_pool,T_pool,tmp

    return Pma_loss , Pmi_loss , Pav_loss
