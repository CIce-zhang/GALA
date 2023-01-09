# -*- coding: utf-8 -*-

'''
Created on  Fri 22 09:50:28 2020

@author: zhangxuefeng

KGA 11.0
'''

import random
import argparse
import time
import gc
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from code.model import Linear_trans,RankGCN
from code.utils import supSGD,load_SRPemb
from code.test import Soft_bt_test, stable_test, test_test, rnm_test
from code.process import data_preprocess, KCore_preprocess_new, KCore_preprocess, load_rnm_data
#from sSGD import supSGD

##########################################################################################
# argument parser
##########################################################################################
opt = dict()

opt['node_num_1'] = 0
opt['rel_num_1'] = 0
opt['node_num_2'] = 0
opt['rel_num_2'] = 0

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--input_dim', dest='input_dim', type=int, default=300,
                    help='-')
parser.add_argument('--output_dim', dest='output_dim', type=int, default=10000,
                    help='-')
parser.add_argument('--augmentation', dest='augmentation', type=str, default="T",
                    help='-')
#training
parser.add_argument('--lr', dest='lr', type=float, default=5e-4,
                    help='-')
parser.add_argument('--optimizer', dest='optimizer', type=str, default="SGD",
                    help='-')
parser.add_argument('--epochs', dest='epochs', type=int, default=30000,
                    help='-')
parser.add_argument('--supSGD', dest='supSGD', type=str, default="T",
                    help='-')
parser.add_argument('--lr_decay', dest='lr_decay', type=str, default="F",
                    help='-')
parser.add_argument('--lr_update', dest='lr_update', type=float, default=0.99,
                    help='-')

parser.add_argument('--pretrain', dest='pretrain', type=str, default="F",
                    help='-')
parser.add_argument('--pretrained_embedding', dest='pretrained_embedding', type=str, default="F",
                    help='-')
parser.add_argument('--pretrained_dir', dest='pretrained_dir', type=str, default="",
                    help='gala_,rand_,or')
parser.add_argument('--save_model', dest='save_model', type=str, default="F",
                    help='-')
parser.add_argument('--save_dir', dest='save_dir', type=str, default="record/tkde_model",
                    help='-')
parser.add_argument('--save_name', dest='save_name', type=str, default="model",
                    help='-')
parser.add_argument('--pre_dir', dest='pre_dir', type=str, default="save",
                    help='dir for pre-train model')

#model
parser.add_argument('--init', dest='init', type=str, default="x_n",
                    help='-')
parser.add_argument('--seed', dest='seed', type=int, default="21",
                    help='-')
parser.add_argument('--gain', dest='gain', type=float, default=0.8,
                    help='-')
parser.add_argument('--rank_pooling', dest='rank_pooling', type=str, default="gmat",
                    help='-')
#dataset
parser.add_argument('--align_num', dest='align_num', type=int, default=15000,
                    help='-')
parser.add_argument('--sup_num', dest='sup_num', type=int, default=0,
                    help='-')
parser.add_argument('--dataset', dest='dataset', type=str, default="../dbp15k/zh_en/mtranse/0_3",
                    help='-')

#hyper-parameter
#   base model
parser.add_argument('--normalize', dest='normalize', type=float, default=60.0,
                    help='')
parser.add_argument('--act', dest='act', type=str, default="relu",
                    help='-')
parser.add_argument('--gcn_num', dest='gcn_num', type=int, default=2,
                    help='-')
#   soft alignment
parser.add_argument('--softalign', dest='softalign', type=str, default="T",
                    help='-')
parser.add_argument('--zoom_type', dest='zoom_type', type=str, default="power",
                    help='-')
parser.add_argument('--zoom_base', dest='zoom_base', type=int, default=1,
                    help='-')
parser.add_argument('--zoom_trunc', dest='zoom_trunc', type=float, default=-0.05,
                    help='0.0 for no trunc')
parser.add_argument('--soft_beta', dest='soft_beta', type=float, default=200.0,
                    help='-')
parser.add_argument('--soft_alpha', dest='soft_alpha', type=float, default=0.95,
                    help='-')
parser.add_argument('--soft_mem', dest='soft_mem', type=float, default=1.0,
                    help='1.0 for not using memory')
#   rank pooling
parser.add_argument('--topk', dest='topk', type=int, default=250,
                    help='-')
parser.add_argument('--div', dest='div', type=int, default=5,
                    help='-')
parser.add_argument('--pw', dest='pw', type=int, default=2000,
                    help='-')
parser.add_argument('--lastk', dest='lastk', type=int, default=250,
                    help='-')
parser.add_argument('--ldiv', dest='ldiv', type=int, default=5,
                    help='-')
parser.add_argument('--lpw', dest='lpw', type=int, default=2000,
                    help='-')
#   flooding
parser.add_argument('--flooding', dest='flooding', type=int, default=500000,
                    help='-1 for no flooding')
#   noise
parser.add_argument('--noise_alpha', dest='noise_alpha', type=int, default=2,
                    help='')
parser.add_argument('--noise_type', dest='noise_type', type=str, default="same_scalar",
                    help='null/vector/same_scalar/dif_scalar')
#   eigenvalue matrix
parser.add_argument('--eigen_vector', dest='eigen_vector', type=str, default="../PR-dbp/d_zh_en",
                    help='-')
parser.add_argument('--eigen_beta', dest='eigen_beta', type=float, default=-0.1,
                    help='-')
#test flag
parser.add_argument('--ablation', dest='ablation', type=str, default="null",
                    help='-')
parser.add_argument('--eigen_adj', dest='eigen_adj', type=str, default="T",
                    help='-')
parser.add_argument('--bias_flag', dest='bias_flag', type=str, default="T",
                    help='-')
parser.add_argument('--relu_flag', dest='relu_flag', type=str, default="T",
                    help='-')
#test
parser.add_argument('--test_freq', dest='test_freq', type=int, default=200,
                    help='-')
parser.add_argument('--h@k', dest='h@k', type=list, default=[1,10,50],
                    help='')
parser.add_argument('--test_metric', dest='test_metric', type=str, default='euclidean',
                    help='euclidean or cityblock or inner')
parser.add_argument('--csls_k', dest='csls_k', type=int, default=2,
                    help='')
#accelate
parser.add_argument('--sub_model', dest='sub_model', type=int, default=1,
                    help='num of model')
parser.add_argument('--shuf_freq', dest='shuf_freq', type=int, default=1,
                    help='-')
#
parser.add_argument('--stable_matching', dest='stable_matching', type=str, default="F",
                    help='-')
parser.add_argument('--max_iteration', dest='max_iteration', type=int, default=100,
                    help='-')
parser.add_argument('--accurate', dest='accurate', type=str, default="F",
                    help='-')
parser.add_argument('--k_core', dest='k_core', type=int, default=1,
                    help='-')

parser.add_argument('--reduce_test', dest='reduce_test', type=str, default="F",
                    help='-')
parser.add_argument('--test_emb', dest='test_emb', type=str, default='raw',
                    help='input or raw')

args = parser.parse_args()
cmd = vars(args)
for k, v in cmd.items():
    if k not in opt or v is not None: opt[k] = v


if opt["lr_update"] > 1:
    opt["lr_update"] = 1/opt["lr_update"]
elif opt["lr_update"] <=0 :
    opt["lr_update"] = 1

    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if opt["seed"] > 0:
    setup_seed(opt["seed"])
else :
    opt["seed"] = random.randrange(0,1000,1)
    setup_seed(opt["seed"])


'''
opt['dataset']='data/dbp15k/zh_en/mtranse/0_3'
opt['eigen_vector']='data/dbp15k/zh_en/mtranse/zh_en'
opt['pretrained_embedding']='T' 
opt['pretrained_dir']='CEA'
'''

#############################################
# def model and functions
#############################################

##########################################################################################
# define inputs
##########################################################################################

##########################################################################################
#####   WARNING

if opt["k_core"] > 1:
    opt["align_num"], opt["sup_num"], opt['node_num_1'], opt['node_num_2'], S_adj, T_adj, ht_1, _, _, _, _, _, S_eye, T_eye, e_dic_1, e_dic_2 = KCore_preprocess(opt['dataset'],opt["k_core"])
else:
    opt["align_num"], opt["sup_num"], opt['node_num_1'], opt['node_num_2'], S_adj, T_adj, _, _, S_eye, T_eye, e_dic_1, e_dic_2 = data_preprocess(opt['dataset'],opt["eigen_adj"],opt["eigen_vector"],opt["eigen_beta"])

if 'SRP' not in opt['dataset']:
    rdata = load_rnm_data(opt['dataset'])

'''
_, _, _, _, _, _, ht_1, _, _, _, _, _, _, _, _, _ = KCore_preprocess_new(opt['dataset'],opt["k_core"])
opt["align_num"], opt["sup_num"], opt['node_num_1'], opt['node_num_2'], S_adj, T_adj, _, _, S_eye, T_eye, e_dic_1, e_dic_2 = data_preprocess(opt['dataset'],opt["eigen_adj"],opt["eigen_vector"],opt["eigen_beta"])
'''
#del ht_1,ht_2
S_input = torch.arange(0,opt['node_num_1'],1).unsqueeze(0).cuda()
T_input = torch.arange(opt['node_num_1'], opt['node_num_1'] + opt['node_num_2'],1).unsqueeze(0).cuda()

Align_test = torch.arange(0,opt['align_num'],1)
if opt["k_core"] > 1:
    kh = []
    for i in ht_1.keys():
        if len(ht_1[i].keys()) > 0 and i < opt['align_num']:
            kh.append(i)
    Align_test = Align_test[kh]

Align_test = Align_test.numpy().tolist()
print("Test list num: %d"%len(Align_test))

for i in range(opt["align_num"],opt["align_num"] + opt["sup_num"]):
    T_input[0][i] = S_input[0][i]

pre_emb = torch.zeros(opt["node_num_1"]+opt["node_num_2"],opt["input_dim"])

#TKDE change:
'''
if opt["pretrained_embedding"] == "T":
    lan = opt["dataset"].find("EN")
    if lan > 0 :
        with open("data/OpenEA/VEC_%s.json"%opt["dataset"][lan:lan+2],"r") as fe:
            se_dic = json.load(fe)
        with open("data/OpenEA/VEC_%s.json"%opt["dataset"][lan+3:lan+5],"r") as fe:
            te_dic = json.load(fe)
        #pre_emb saved pretrained embeddings from faxtText
        for e in e_dic_1.keys():
            pre_emb[e_dic_1[e]] = torch.tensor(se_dic[e])
        for e in e_dic_2.keys():
            pre_emb[e_dic_2[e]+opt["node_num_1"]] = torch.tensor(te_dic[e])
'''
'''
with open(tmp_dir, mode='r', encoding='utf-8') as f:
    embedding_list = json.load(f)
    print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    ne_vec = np.array(embedding_list)
'''

if opt["pretrained_embedding"] == "T":
    if '15k' not in opt['pretrained_dir']:
        lang = opt["dataset"].split('/')[2]
        assert 'en' in lang
        #with open('../KGAtest/rnm_data/DBP15k/'+lang+'/vectorList.json','r') as f:
        #with open('../KGAtest/rnm_data/DBP15k/'+lang+'/rand_vectorList.json','r') as f:
        #with open('../KGAtest/rnm_data/DBP15k/'+lang+'/gala_vectorList.json','r') as f:
        if 'CEA' in opt['pretrained_dir']:
            tmp_dir = opt['pretrained_dir'] + '/data/' + lang + '/' + lang.split('_')[0] + '_vectorList.json'
            with open(tmp_dir, mode='r', encoding='utf-8') as f:
                d = json.load(f)
        else:
            tmp_dir = '../KGAtest/rnm_data/DBP15k/'+lang+'/' + opt['pretrained_dir'] +'vectorList.json'
            with open(tmp_dir,'r') as f:
                d=json.load(f)
        #pre_emb = torch.tensor(d)
        if 'gala' in opt['pretrained_dir']:
        #if True:
            pre_emb = torch.tensor(d)
            print("load embedding directly")
        else:
            for e in e_dic_1.keys():
                pre_emb[e_dic_1[e]] = torch.tensor(d[int(e)])
            for e in e_dic_2.keys():
                pre_emb[e_dic_2[e]+opt["node_num_1"]] = torch.tensor(d[int(e)])
            assert len(e_dic_1) + len(e_dic_2) == len(pre_emb)
        print("load pretrained models from "+tmp_dir)
        del d
        #此处设计时，通过e_dic_1/2对所有实体进行重新编码了，且每个KG都是从0开始的编码
        #因此，需要逐行提取预训练嵌入
        #目测：rand_vectorList文件当中应该是每行的行号和每行的向量相对应，行号决定了是哪一个实体
    else:
        d = load_SRPemb(opt['pretrained_dir']+'/name_vec.txt')
        for e in e_dic_1.keys():
            pre_emb[e_dic_1[e]] = torch.tensor(d[int(e)])
        for e in e_dic_2.keys():
            pre_emb[e_dic_2[e]+opt["node_num_1"]] = torch.tensor(d[int(e)])
        assert len(e_dic_1) + len(e_dic_2) == len(pre_emb)
        del d




'''
with open('tempdic1.json','w') as f:
    json.dump(e_dic_1,f)
with open('tempdic2.json','w') as f:
    json.dump(e_dic_2,f)
#torch.save({'e1':e_dic_1,'e2':e_dic_2},'temp')
with open('tempdic1.json','r') as f:
    enti1=json.load(f)

with open('tempdic2.json','r') as f:
    enti2=json.load(f)
print('dic saved')
'''
#python -u train.py --dataset data/OpenEA/EN_DE_15K_V2 --eigen_adj F --test_freq 20 --pretrained_embedding F
#直接把embedding取出来，再送进net进行复制？
#还是说把embedding传入net，net中索引再来弄？
#e_dic:把原始名称映射到对应数字上
#e_dic_1 = {v,k for k,v in e_dic_1.items()}

Align_anchor = []
start_epoch = 0
anchors = []
anchor_set =set(anchors)
checks = []
ac_diss = []
anchor_test = [0]*(len(opt["h@k"])+2)
data_list = list(range(opt["sub_model"]))

s_weight=torch.ones(opt['node_num_1']).cuda()
t_weight=torch.ones(opt['node_num_2']).cuda()

s_weight[opt["align_num"]:opt["align_num"] + opt["sup_num"]] = opt["normalize"]
t_weight[opt["align_num"]:opt["align_num"] + opt["sup_num"]] = opt["normalize"]
##########################################################################################
# build model
##########################################################################################

LT_model = Linear_trans(opt,pre_emb).cuda()
del pre_emb
####!!!!
'''
if opt['pretrained_embedding'] == 'T':
    print("!!!!!!!!!!WARNING: reload from local model")
    lang = opt["dataset"].split('/')[2]
    LT_model.load_state_dict(torch.load('Model_'+lang)) 
'''

GCN_model_list = [RankGCN(opt).cuda() for _ in range(opt["sub_model"])]
# define optimizer 
parameters = [p for p in LT_model.parameters() if p.requires_grad]
for i in range(opt["sub_model"]):
    for p in GCN_model_list[i].parameters():
        if p.requires_grad:
            parameters.append(p)

if opt["supSGD"] == "T":
    a_optim=supSGD(parameters, lr=opt['lr'], weight_decay=0.)
else :
    if opt["optimizer"] == "Adamax":
        optimizer = torch.optim.Adamax(parameters, lr=opt['lr'], weight_decay=0.)
    elif opt["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(parameters, lr=opt['lr'], weight_decay=0.)

if opt["save_model"] == "T" and not os.path.exists(opt["save_dir"]):
    os.makedirs(opt["save_dir"])

LT_model.train()
for model in GCN_model_list:
    model.train()
print(opt)

'''
for para in LT_model.parameters():
    if para.requires_grad == True:
        print(para.size())

for para in GCN_model_list[0].parameters():
    if para.requires_grad == True:
        print(para.size())

'''
##########################################################################################
#train & test
##########################################################################################
print("begin training")

maxepo = -1
maxacc = -1
maxepo_rnm = -1
maxacc_rnm = -1
f = open("%s.txt"%opt["save_dir"],"a")

so_mem = torch.zeros(1)
to_mem = torch.zeros(1)

if 'SRP' not in opt['dataset']:
    maxacc_rnm, maxepo_rnm = rnm_test(LT_model, opt, 0, S_input.squeeze(0), T_input.squeeze(0), Align_test, maxacc_rnm, maxepo_rnm, rdata)
if opt["stable_matching"] =="T":
    maxacc, maxepo, _, _,so_mem, to_mem, _, _ = stable_test(LT_model, opt, 0, S_input.squeeze(0), T_input.squeeze(0), S_adj, T_adj, so_mem, to_mem, Align_test, maxacc, maxepo,f,anchor_test)
else :
    maxacc, maxepo, _, _,so_mem, to_mem, _, _ = Soft_bt_test(LT_model, opt, 0, S_input.squeeze(0), T_input.squeeze(0), S_adj, T_adj, so_mem, to_mem, Align_test, maxacc, maxepo,f,anchor_test)


f.close()
firstacc = maxacc

tr_time = time.time()

#TKDE change
'''
if opt["pretrained_embedding"] == "T":
    lang = opt["dataset"].split('/')[2]
    torch.save(LT_model.state_dict(),'Model_'+lang+'_pre_'+opt['pretrained_dir'])
else:
    lang = opt["dataset"].split('/')[2]
    torch.save(LT_model.state_dict(),'Model_'+lang)
'''

if opt["save_model"] == "T" and opt["pretrain"] != "T" :
        LT_model.eval()
        if opt["supSGD"] == "T":
            state = {"model":LT_model.state_dict(), "epoch":i+1, "anchor_test":anchor_test}
            #torch.save(state,"%s/Model_%s_Epo%d"%(opt["save_dir"],opt["dataset"][opt["dataset"].rfind("/")+1:], 0))
            torch.save(state,"%s/Model_%s_Epo%d"%(opt["save_dir"],opt["save_name"], 0))


for i in range(opt['epochs']):
    # forward 
    f = open("%s.txt"%opt["save_dir"],"a")
    i += start_epoch
    max_loss = []
    min_loss = []
    mean_loss = []
    ra1_loss = []
    ra2_loss = []
    A_loss = []
    loss_stat = torch.zeros(6)

    t_time=time.time()
    if i%opt["shuf_freq"] == 0 :
        random.shuffle(data_list)
    for j in data_list:
        g_time = time.time()
        TA_loss = 0
        LT_model.train()
        LT_model.zero_grad()
        S_o, T_o = LT_model.forward(S_input, T_input, [np.array(anchors).reshape(-1),np.array(checks).reshape(-1)],s_weight, t_weight)
        GCN_model_list[j].train()
        GCN_model_list[j].zero_grad()
        max_loss, min_loss, mean_loss, ra1_loss, ra2_loss = GCN_model_list[j].forward(S_adj, T_adj, S_eye, T_eye, S_o[:,:,int(j*opt["output_dim"]/opt["sub_model"]):int((j+1)*opt["output_dim"]/opt["sub_model"])], T_o[:,:,int(j*opt["output_dim"]/opt["sub_model"]):int((j+1)*opt["output_dim"]/opt["sub_model"])])
        for  jj in range(opt["gcn_num"]):
            TA_loss += max_loss[jj] + min_loss[jj] + mean_loss[jj] + ra1_loss[jj] + ra2_loss[jj]
            A_loss.append(TA_loss.clone().detach())
            loss_stat[1] += max_loss[jj].clone().detach()
            loss_stat[2] += min_loss[jj].clone().detach()
            loss_stat[3] += mean_loss[jj].clone().detach()
            loss_stat[4] += ra1_loss[jj].clone().detach()
            loss_stat[5] += ra2_loss[jj].clone().detach()
        loss_stat[0] += A_loss[-1]
        if opt["flooding"] > 0:
            TA_loss = torch.abs(TA_loss-opt["flooding"]) + opt["flooding"]
        TA_loss.backward()
        print("Batch %d: time: %.2f | total loss :%.5f "%(j,time.time()-g_time,A_loss[-1]))
        if opt["supSGD"] == "T":
            a_optim.step()
        else :
            optimizer.step()
    
    if opt["lr_decay"] == "T" and opt["supSGD"] == "T":
        a_optim.update_lr(opt["lr_update"])
    print("Epoch {} : time {:.2f} | loss(total|max|min|mean|TopK|LastK|Cov): {}".format(i+1,time.time()-t_time, loss_stat.numpy().tolist()))

    if (i+1)%opt["test_freq"] == 0 and i > 45 :
        with torch.no_grad():
            LT_model.eval()
            if 'SRP' not in opt['dataset']:
                maxacc_rnm, maxepo_rnm = rnm_test(LT_model, opt, i+1, S_input.squeeze(0), T_input.squeeze(0), Align_test, maxacc_rnm, maxepo_rnm, rdata)
            if opt["stable_matching"] =="T":
                maxacc, maxepo, so, to, so_mem, to_mem, s_weight, t_weight = stable_test(LT_model, opt, i+1, S_input.squeeze(0), T_input.squeeze(0), S_adj, T_adj, so_mem, to_mem, Align_test, maxacc, maxepo,f,anchor_test)
            else :
                maxacc, maxepo, so, to, so_mem, to_mem, s_weight, t_weight = Soft_bt_test(LT_model, opt, i+1, S_input.squeeze(0), T_input.squeeze(0), S_adj, T_adj, so_mem, to_mem, Align_test, maxacc, maxepo,f,anchor_test)
            if opt["softalign"] == "T":
                LT_model.Eemb.weight[T_input[0]] = to
                LT_model.Eemb.weight[S_input[0]] = so
            del so, to
            LT_model = LT_model.cuda()
    tr_time = time.time()
    if (i+1)%1000 == 0 and opt['save_model'] == "T":
        print("save model...")
        f.write("save model...\n")
        LT_model.eval()
        if opt["supSGD"] == "T":
            state = {"model":LT_model.state_dict(), "epoch":i+1, "anchor_test":anchor_test}
            #torch.save(state,"%s/Model_%s_Epo%d"%(opt["save_dir"],opt["dataset"][opt["dataset"].rfind("/")+1:], i+1))
            torch.save(state,"%s/Model_%s_Epo%d"%(opt["save_dir"],opt["save_name"], i+1))
    f.close()

print("end training")
print(maxacc)
print(maxepo)
print(firstacc)


