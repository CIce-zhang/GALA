import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def data_preprocess(data = "zh_en", eigen_flag = "F", eigen_vec = "", beta = 0):
    if eigen_flag == "T":
        return eigen_data_preprocess(data,eigen_vec, beta)
    if data.find("OpenEA") > 0:
        return DW_preprocess_new(data)
    else :
        return DW_preprocess(data)


def KCore_preprocess_new(data = "D_W_15K_V1", k_core = 1):
    ent_dic_1 = {}
    ent_dic_2 = {}
    num_ent_1 = 0
    num_ent_2 = 0
    with open(data+"/721_5fold/1/test_links", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            ent_dic_1[d[i][0]]=num_ent_1
            ent_dic_2[d[i][1]]=num_ent_2
            num_ent_1 += 1
            num_ent_2 += 1
        ref = len(d)
    with open(data+"/721_5fold/1/valid_links", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            ent_dic_1[d[i][0]]=num_ent_1
            ent_dic_2[d[i][1]]=num_ent_2
            num_ent_1 += 1
            num_ent_2 += 1
        valid = len(d)
    with open(data+"/721_5fold/1/train_links", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            ent_dic_1[d[i][0]]=num_ent_1
            ent_dic_2[d[i][1]]=num_ent_2
            num_ent_1 += 1
            num_ent_2 += 1
        sup = len(d)
    
    with open(data+"/ent_links", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            if d[i][0] not in ent_dic_1:
                ent_dic_1[d[i][0]] = num_ent_1
                num_ent_1 += 1
            if d[i][1] not in ent_dic_2:
                ent_dic_2[d[i][1]] = num_ent_2
                num_ent_2 += 1
    
    with open(data+"/rel_triples_1", "r", encoding = "utf-8") as f:
        d = f.readlines()
        s_i = [[],[]]
        s_v = []
        eye_i = [[],[]]
        eye_v = []
        h_t = dict([x,dict()] for x in range(num_ent_1))
        #finally,h_t is the k_core result and kh_t is the rest result  
        
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            d[i][0] = ent_dic_1[d[i][0]]
            d[i][2] = ent_dic_1[d[i][2]]
            
            if d[i][0] == d[i][2]:
                continue
            if d[i][2] not in h_t[d[i][0]].keys():
                h_t[d[i][0]][d[i][2]] = 1
            else:
                h_t[d[i][0]][d[i][2]] += 1
            if d[i][0] not in h_t[d[i][2]].keys():
                h_t[d[i][2]][d[i][0]] = 1
            else:
                h_t[d[i][2]][d[i][0]] += 1
        # sparse adj for entity&entity:
        # undirect single graph
        ks_i = [[],[]]
        ks_v = []
        kh_t = dict([x,dict()] for x in range(num_ent_1))
        #finally,h_t is the k_core result and kh_t is the rest result  
        count = True
        #opt["kcore"]
        #if self_loop=True,k=k+1
        #NO self-loop in h_t
        while count:
            count = False
            for i in range(num_ent_1):
                if len(h_t[i].keys()) < k_core:
                    conut = True
                    for j in list(h_t[i].keys()):
                        kh_t[i][j] = h_t[i][j]
                        kh_t[j][i] = h_t[j][i]
                        del h_t[i][j], h_t[j][i]
        for i in range(num_ent_1):
            tmpv = 1/(len(kh_t[i].keys())+1)
            #tmpv=1
            for j in kh_t[i].keys():
                ks_i[0].append(i)
                ks_i[1].append(j)
                ks_v.append(tmpv)
            ks_i[0].append(i)
            ks_i[1].append(i)
            ks_v.append(tmpv)
        ks_i = torch.LongTensor(ks_i)
        ks_v = torch.FloatTensor(ks_v)
        KS_adj_1 = torch.sparse.FloatTensor(ks_i, ks_v, torch.Size((num_ent_1, num_ent_1))).cuda()
        h_t1 = h_t
        kh_t1 = kh_t
        
        for i in range(num_ent_1):
            tmpv = 1/(len(h_t[i].keys())+1)
            #tmpv=1
            for j in h_t[i].keys():
                s_i[0].append(i)
                s_i[1].append(j)
                s_v.append(tmpv)
            s_i[0].append(i)
            s_i[1].append(i)
            s_v.append(tmpv)
            eye_i[0].append(i)
            eye_i[1].append(i)
            eye_v.append(tmpv)
        eye_i = torch.LongTensor(eye_i)
        eye_v = torch.FloatTensor(eye_v)
        S_eye = torch.sparse.FloatTensor(eye_i, eye_v, torch.Size((num_ent_1, num_ent_1))).cuda()
        
        s_i = torch.LongTensor(s_i)
        s_v = torch.FloatTensor(s_v)
        S_adj_1 = torch.sparse.FloatTensor(s_i, s_v, torch.Size((num_ent_1, num_ent_1))).cuda()
        h_t1 = h_t

    with open(data+"/rel_triples_2", "r", encoding = "utf-8") as f:
        d = f.readlines()
        s_i = [[],[]]
        s_v = []
        eye_i = [[],[]]
        eye_v = []
        h_t = dict([x,dict()] for x in range(num_ent_2))
        
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            d[i][0] = ent_dic_2[d[i][0]]
            d[i][2] = ent_dic_2[d[i][2]]
            
            if d[i][0] == d[i][2]:
                continue
            if d[i][2] not in h_t[d[i][0]].keys():
                h_t[d[i][0]][d[i][2]] = 1
            else:
                h_t[d[i][0]][d[i][2]] += 1
            if d[i][0] not in h_t[d[i][2]].keys():
                h_t[d[i][2]][d[i][0]] = 1
            else:
                h_t[d[i][2]][d[i][0]] += 1
        
        ks_i = [[],[]]
        ks_v = []
        kh_t = dict([x,dict()] for x in range(num_ent_2))
        #finally,h_t is the k_core result and kh_t is the rest result  
        count = True
        #opt["kcore"]
        #if self_loop=True,k=k+1
        #NO self-loop in h_t
        while count:
            count = False
            for i in range(num_ent_2):
                if len(h_t[i].keys()) < k_core:
                    conut = True
                    for j in list(h_t[i].keys()):
                        kh_t[i][j] = h_t[i][j]
                        kh_t[j][i] = h_t[j][i]
                        del h_t[i][j], h_t[j][i]
        for i in range(num_ent_2):
            tmpv = 1/(len(kh_t[i].keys())+1)
            #tmpv=1
            for j in kh_t[i].keys():
                ks_i[0].append(i)
                ks_i[1].append(j)
                ks_v.append(tmpv)
            ks_i[0].append(i)
            ks_i[1].append(i)
            ks_v.append(tmpv)
        ks_i = torch.LongTensor(ks_i)
        ks_v = torch.FloatTensor(ks_v)
        KS_adj_2 = torch.sparse.FloatTensor(ks_i, ks_v, torch.Size((num_ent_2, num_ent_2))).cuda()
        h_t2 = h_t
        kh_t2 = kh_t
        
        for i in range(num_ent_2):
            tmpv = 1/(len(h_t[i].keys())+1)
            #tmpv=1
            for j in h_t[i].keys():
                s_i[0].append(i)
                s_i[1].append(j)
                s_v.append(tmpv)
            s_i[0].append(i)
            s_i[1].append(i)
            s_v.append(tmpv)
            eye_i[0].append(i)
            eye_i[1].append(i)
            eye_v.append(tmpv)
        eye_i = torch.LongTensor(eye_i)
        eye_v = torch.FloatTensor(eye_v)
        T_eye = torch.sparse.FloatTensor(eye_i, eye_v, torch.Size((num_ent_2, num_ent_2))).cuda()
        
        s_i = torch.LongTensor(s_i)
        s_v = torch.FloatTensor(s_v)
        S_adj_2 = torch.sparse.FloatTensor(s_i, s_v, torch.Size((num_ent_2,num_ent_2))).cuda()
        h_t2 = h_t

    del s_i, s_v, h_t, d
    #, KS_adj_1, KS_adj_2
    return ref+valid, sup, num_ent_1, num_ent_2, S_adj_1, S_adj_2, h_t1, h_t2, KS_adj_1, KS_adj_2, kh_t1, kh_t2,S_eye, T_eye, ent_dic_1, ent_dic_2

def KCore_preprocess(data = "D_W_15K_V1", k_core = 1):
    ent_dic_1 = {}
    ent_dic_2 = {}
    num_ent_1 = 0
    num_ent_2 = 0
    with open(data+"/ref_pairs", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            for j in range(len(d[i])):
                d[i][j] = int(d[i][j])
            ent_dic_1[d[i][0]]=num_ent_1
            ent_dic_2[d[i][1]]=num_ent_2
            num_ent_1 += 1
            num_ent_2 += 1
        ref = len(d)
    with open(data+"/sup_pairs", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            for j in range(len(d[i])):
                d[i][j] = int(d[i][j])
            ent_dic_1[d[i][0]]=num_ent_1
            ent_dic_2[d[i][1]]=num_ent_2
            num_ent_1 += 1
            num_ent_2 += 1
        sup = len(d)

    with open(data+"/ent_ids_1", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            tmp = int(d[i][0])
            if tmp not in ent_dic_1:
                ent_dic_1[tmp] = num_ent_1
                num_ent_1 += 1

    with open(data+"/ent_ids_2", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            tmp = int(d[i][0])
            if tmp not in ent_dic_2:
                ent_dic_2[tmp] = num_ent_2
                num_ent_2 += 1
    
    with open(data+"/triples_1", "r", encoding = "utf-8") as f:
        d = f.readlines()
        s_i = [[],[]]
        s_v = []
        eye_i = [[],[]]
        eye_v = []
        h_t = dict([x,dict()] for x in range(num_ent_1))
        #finally,h_t is the k_core result and kh_t is the rest result  
        
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            d[i][0] = ent_dic_1[int(d[i][0])]
            d[i][2] = ent_dic_1[int(d[i][2])]
            
            if d[i][0] == d[i][2]:
                continue
            if d[i][2] not in h_t[d[i][0]].keys():
                h_t[d[i][0]][d[i][2]] = 1
            else:
                h_t[d[i][0]][d[i][2]] += 1
            if d[i][0] not in h_t[d[i][2]].keys():
                h_t[d[i][2]][d[i][0]] = 1
            else:
                h_t[d[i][2]][d[i][0]] += 1
        # sparse adj for entity&entity:
        # undirect single graph
        ks_i = [[],[]]
        ks_v = []
        kh_t = dict([x,dict()] for x in range(num_ent_1))
        #finally,h_t is the k_core result and kh_t is the rest result  
        count = True
        #opt["kcore"]
        #if self_loop=True,k=k+1
        #NO self-loop in h_t
        while count:
            count = False
            for i in range(num_ent_1):
                if len(h_t[i].keys()) < k_core:
                    conut = True
                    for j in list(h_t[i].keys()):
                        kh_t[i][j] = h_t[i][j]
                        kh_t[j][i] = h_t[j][i]
                        del h_t[i][j], h_t[j][i]
        for i in range(num_ent_1):
            tmpv = 1/(len(kh_t[i].keys())+1)
            #tmpv=1
            for j in kh_t[i].keys():
                ks_i[0].append(i)
                ks_i[1].append(j)
                ks_v.append(tmpv)
            ks_i[0].append(i)
            ks_i[1].append(i)
            ks_v.append(tmpv)
        ks_i = torch.LongTensor(ks_i)
        ks_v = torch.FloatTensor(ks_v)
        KS_adj_1 = torch.sparse.FloatTensor(ks_i, ks_v, torch.Size((num_ent_1, num_ent_1))).cuda()
        h_t1 = h_t
        kh_t1 = kh_t
        
        for i in range(num_ent_1):
            tmpv = 1/(len(h_t[i].keys())+1)
            #tmpv=1
            for j in h_t[i].keys():
                s_i[0].append(i)
                s_i[1].append(j)
                s_v.append(tmpv)
            s_i[0].append(i)
            s_i[1].append(i)
            s_v.append(tmpv)
            eye_i[0].append(i)
            eye_i[1].append(i)
            eye_v.append(tmpv)
        eye_i = torch.LongTensor(eye_i)
        eye_v = torch.FloatTensor(eye_v)
        S_eye = torch.sparse.FloatTensor(eye_i, eye_v, torch.Size((num_ent_1, num_ent_1))).cuda()
        
        s_i = torch.LongTensor(s_i)
        s_v = torch.FloatTensor(s_v)
        S_adj_1 = torch.sparse.FloatTensor(s_i, s_v, torch.Size((num_ent_1, num_ent_1))).cuda()
        h_t1 = h_t

    with open(data+"/triples_2", "r", encoding = "utf-8") as f:
        d = f.readlines()
        s_i = [[],[]]
        s_v = []
        eye_i = [[],[]]
        eye_v = []
        h_t = dict([x,dict()] for x in range(num_ent_2))
        
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            d[i][0] = ent_dic_2[int(d[i][0])]
            d[i][2] = ent_dic_2[int(d[i][2])]
            
            if d[i][0] == d[i][2]:
                continue
            if d[i][2] not in h_t[d[i][0]].keys():
                h_t[d[i][0]][d[i][2]] = 1
            else:
                h_t[d[i][0]][d[i][2]] += 1
            if d[i][0] not in h_t[d[i][2]].keys():
                h_t[d[i][2]][d[i][0]] = 1
            else:
                h_t[d[i][2]][d[i][0]] += 1
        
        ks_i = [[],[]]
        ks_v = []
        kh_t = dict([x,dict()] for x in range(num_ent_2))
        #finally,h_t is the k_core result and kh_t is the rest result  
        count = True
        #opt["kcore"]
        #if self_loop=True,k=k+1
        #NO self-loop in h_t
        while count:
            count = False
            for i in range(num_ent_2):
                if len(h_t[i].keys()) < k_core:
                    conut = True
                    for j in list(h_t[i].keys()):
                        kh_t[i][j] = h_t[i][j]
                        kh_t[j][i] = h_t[j][i]
                        del h_t[i][j], h_t[j][i]
        for i in range(num_ent_2):
            tmpv = 1/(len(kh_t[i].keys())+1)
            #tmpv=1
            for j in kh_t[i].keys():
                ks_i[0].append(i)
                ks_i[1].append(j)
                ks_v.append(tmpv)
            ks_i[0].append(i)
            ks_i[1].append(i)
            ks_v.append(tmpv)
        ks_i = torch.LongTensor(ks_i)
        ks_v = torch.FloatTensor(ks_v)
        KS_adj_2 = torch.sparse.FloatTensor(ks_i, ks_v, torch.Size((num_ent_2, num_ent_2))).cuda()
        h_t2 = h_t
        kh_t2 = kh_t
        
        for i in range(num_ent_2):
            tmpv = 1/(len(h_t[i].keys())+1)
            #tmpv=1
            for j in h_t[i].keys():
                s_i[0].append(i)
                s_i[1].append(j)
                s_v.append(tmpv)
            s_i[0].append(i)
            s_i[1].append(i)
            s_v.append(tmpv)
            eye_i[0].append(i)
            eye_i[1].append(i)
            eye_v.append(tmpv)
        eye_i = torch.LongTensor(eye_i)
        eye_v = torch.FloatTensor(eye_v)
        T_eye = torch.sparse.FloatTensor(eye_i, eye_v, torch.Size((num_ent_2, num_ent_2))).cuda()
        
        s_i = torch.LongTensor(s_i)
        s_v = torch.FloatTensor(s_v)
        S_adj_2 = torch.sparse.FloatTensor(s_i, s_v, torch.Size((num_ent_2,num_ent_2))).cuda()
        h_t2 = h_t

    del s_i, s_v, h_t, d
    #, KS_adj_1, KS_adj_2
    return ref, sup, num_ent_1, num_ent_2, S_adj_1, S_adj_2, h_t1, h_t2, KS_adj_1, KS_adj_2, kh_t1, kh_t2,S_eye, T_eye, ent_dic_1, ent_dic_2

def DW_preprocess_new(data = "D_W_15K_V1"):
    ent_dic_1 = {}
    ent_dic_2 = {}
    num_ent_1 = 0
    num_ent_2 = 0
    with open(data+"/721_5fold/1/test_links", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            ent_dic_1[d[i][0]]=num_ent_1
            ent_dic_2[d[i][1]]=num_ent_2
            num_ent_1 += 1
            num_ent_2 += 1
        ref = len(d)
    with open(data+"/721_5fold/1/valid_links", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            ent_dic_1[d[i][0]]=num_ent_1
            ent_dic_2[d[i][1]]=num_ent_2
            num_ent_1 += 1
            num_ent_2 += 1
        valid = len(d)
    with open(data+"/721_5fold/1/train_links", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            ent_dic_1[d[i][0]]=num_ent_1
            ent_dic_2[d[i][1]]=num_ent_2
            num_ent_1 += 1
            num_ent_2 += 1
        sup = len(d)
    
    
    with open(data+"/ent_links", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            if d[i][0] not in ent_dic_1:
                ent_dic_1[d[i][0]] = num_ent_1
                num_ent_1 += 1
            if d[i][1] not in ent_dic_2:
                ent_dic_2[d[i][1]] = num_ent_2
                num_ent_2 += 1
    
    with open(data+"/rel_triples_1", "r", encoding = "utf-8") as f:
        d = f.readlines()
        s_i = [[],[]]
        s_v = []
        eye_i = [[],[]]
        eye_v = []
        h_t = dict([x,dict()] for x in range(num_ent_1))
        
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            d[i][0] = ent_dic_1[d[i][0]]
            d[i][2] = ent_dic_1[d[i][2]]
            
            if d[i][0] == d[i][2]:
                continue
            if d[i][2] not in h_t[d[i][0]].keys():
                h_t[d[i][0]][d[i][2]] = 1
            else:
                h_t[d[i][0]][d[i][2]] += 1
            if d[i][0] not in h_t[d[i][2]].keys():
                h_t[d[i][2]][d[i][0]] = 1
            else:
                h_t[d[i][2]][d[i][0]] += 1
        # sparse adj for entity&entity:
        # undirect single graph
        for i in range(num_ent_1):
            tmpv = 1/(len(h_t[i].keys())+1)
            #tmpv=1
            for j in h_t[i].keys():
                s_i[0].append(i)
                s_i[1].append(j)
                s_v.append(tmpv)
            s_i[0].append(i)
            s_i[1].append(i)
            eye_i[0].append(i)
            eye_i[1].append(i)
            s_v.append(tmpv)
            eye_v.append(tmpv)
        eye_i = torch.LongTensor(eye_i)
        eye_v = torch.FloatTensor(eye_v)
        S_eye = torch.sparse.FloatTensor(eye_i, eye_v, torch.Size((num_ent_1, num_ent_1))).cuda()
    
        s_i = torch.LongTensor(s_i)
        s_v = torch.FloatTensor(s_v)
        S_adj_1 = torch.sparse.FloatTensor(s_i, s_v, torch.Size((num_ent_1, num_ent_1))).cuda()
        h_t1 = h_t

    with open(data+"/rel_triples_2", "r", encoding = "utf-8") as f:
        d = f.readlines()
        s_i = [[],[]]
        s_v = []
        eye_i = [[],[]]
        eye_v = []
        h_t = dict([x,dict()] for x in range(num_ent_2))
        
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            d[i][0] = ent_dic_2[d[i][0]]
            d[i][2] = ent_dic_2[d[i][2]]
            
            if d[i][0] == d[i][2]:
                continue
            if d[i][2] not in h_t[d[i][0]].keys():
                h_t[d[i][0]][d[i][2]] = 1
            else:
                h_t[d[i][0]][d[i][2]] += 1
            if d[i][0] not in h_t[d[i][2]].keys():
                h_t[d[i][2]][d[i][0]] = 1
            else:
                h_t[d[i][2]][d[i][0]] += 1
        
        for i in range(num_ent_2):
            tmpv = 1/(len(h_t[i].keys())+1)
            #tmpv=1
            for j in h_t[i].keys():
                s_i[0].append(i)
                s_i[1].append(j)
                s_v.append(tmpv)
            s_i[0].append(i)
            s_i[1].append(i)
            eye_i[0].append(i)
            eye_i[1].append(i)
            s_v.append(tmpv)
            eye_v.append(tmpv)
        eye_i = torch.LongTensor(eye_i)
        eye_v = torch.FloatTensor(eye_v)
        T_eye = torch.sparse.FloatTensor(eye_i, eye_v, torch.Size((num_ent_2, num_ent_2))).cuda()
        
        s_i = torch.LongTensor(s_i)
        s_v = torch.FloatTensor(s_v)
        S_adj_2 = torch.sparse.FloatTensor(s_i, s_v, torch.Size((num_ent_2,num_ent_2))).cuda()
        h_t2 = h_t

    del s_i, s_v, h_t, d
    return ref+valid, sup, num_ent_1, num_ent_2, S_adj_1, S_adj_2, h_t1, h_t2, S_eye, T_eye, ent_dic_1, ent_dic_2

def DW_preprocess(data = "zh_en"):
    ent_dic_1 = {}
    ent_dic_2 = {}
    num_ent_1 = 0
    num_ent_2 = 0
    with open(data+"/ref_pairs", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            for j in range(len(d[i])):
                d[i][j] = int(d[i][j])
            ent_dic_1[d[i][0]]=num_ent_1
            ent_dic_2[d[i][1]]=num_ent_2
            num_ent_1 += 1
            num_ent_2 += 1
        ref = len(d)
    with open(data+"/sup_pairs", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            for j in range(len(d[i])):
                d[i][j] = int(d[i][j])
            ent_dic_1[d[i][0]]=num_ent_1
            ent_dic_2[d[i][1]]=num_ent_2
            num_ent_1 += 1
            num_ent_2 += 1
        sup = len(d)

    with open(data+"/ent_ids_1", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            tmp = int(d[i][0])
            if tmp not in ent_dic_1:
                ent_dic_1[tmp] = num_ent_1
                num_ent_1 += 1

    with open(data+"/ent_ids_2", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            tmp = int(d[i][0])
            if tmp not in ent_dic_2:
                ent_dic_2[tmp] = num_ent_2
                num_ent_2 += 1
    
    with open(data+"/triples_1", "r", encoding = "utf-8") as f:
        d = f.readlines()
        s_i = [[],[]]
        s_v = []
        eye_i = [[],[]]
        eye_v = []
        h_t = dict([x,dict()] for x in range(num_ent_1))
        
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            d[i][0] = ent_dic_1[int(d[i][0])]
            d[i][2] = ent_dic_1[int(d[i][2])]
            
            if d[i][0] == d[i][2]:
                continue
            if d[i][2] not in h_t[d[i][0]].keys():
                h_t[d[i][0]][d[i][2]] = 1
            else:
                h_t[d[i][0]][d[i][2]] += 1
            if d[i][0] not in h_t[d[i][2]].keys():
                h_t[d[i][2]][d[i][0]] = 1
            else:
                h_t[d[i][2]][d[i][0]] += 1
    
        # sparse adj for entity&entity:
        # undirect single graph
        for i in range(num_ent_1):
            tmpv = 1/(len(h_t[i].keys())+1)
            for j in h_t[i].keys():
                s_i[0].append(i)
                s_i[1].append(j)
                s_v.append(tmpv)
            s_i[0].append(i)
            s_i[1].append(i)
            eye_i[0].append(i)
            eye_i[1].append(i)
            s_v.append(tmpv)
            eye_v.append(tmpv)
        eye_i = torch.LongTensor(eye_i)
        eye_v = torch.FloatTensor(eye_v)
        S_eye = torch.sparse.FloatTensor(eye_i, eye_v, torch.Size((num_ent_1, num_ent_1))).cuda()

        s_i = torch.LongTensor(s_i)
        s_v = torch.FloatTensor(s_v)
        S_adj_1 = torch.sparse.FloatTensor(s_i, s_v, torch.Size((num_ent_1, num_ent_1))).cuda()
        h_t1 = h_t

    with open(data+"/triples_2", "r", encoding = "utf-8") as f:
        d = f.readlines()
        s_i = [[],[]]
        s_v = []
        eye_i = [[],[]]
        eye_v = []
        h_t = dict([x,dict()] for x in range(num_ent_2))
        
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            d[i][0] = ent_dic_2[int(d[i][0])]
            d[i][2] = ent_dic_2[int(d[i][2])]
            
            if d[i][0] == d[i][2]:
                continue
            if d[i][2] not in h_t[d[i][0]].keys():
                h_t[d[i][0]][d[i][2]] = 1
            else:
                h_t[d[i][0]][d[i][2]] += 1
            if d[i][0] not in h_t[d[i][2]].keys():
                h_t[d[i][2]][d[i][0]] = 1
            else:
                h_t[d[i][2]][d[i][0]] += 1
        
        for i in range(num_ent_2):
            tmpv = 1/(len(h_t[i].keys())+1)
            for j in h_t[i].keys():
                s_i[0].append(i)
                s_i[1].append(j)
                s_v.append(tmpv)
            s_i[0].append(i)
            s_i[1].append(i)
            eye_i[0].append(i)
            eye_i[1].append(i)
            s_v.append(tmpv)
            eye_v.append(tmpv)
        eye_i = torch.LongTensor(eye_i)
        eye_v = torch.FloatTensor(eye_v)
        T_eye = torch.sparse.FloatTensor(eye_i, eye_v, torch.Size((num_ent_2, num_ent_2))).cuda()
        
        s_i = torch.LongTensor(s_i)
        s_v = torch.FloatTensor(s_v)
        S_adj_2 = torch.sparse.FloatTensor(s_i, s_v, torch.Size((num_ent_2,num_ent_2))).cuda()
        h_t2 = h_t


    del s_i, s_v, h_t, d
    return ref, sup, num_ent_1, num_ent_2, S_adj_1, S_adj_2, h_t1, h_t2, S_eye, T_eye, ent_dic_1, ent_dic_2

def eigen_data_preprocess(data = "zh_en",eigen_vec = "", beta = 0):
    ent_dic_1 = {}
    ent_dic_2 = {}
    num_ent_1 = 0
    num_ent_2 = 0
    e_v_1 = np.load(eigen_vec+"_0.npy").squeeze()
    e_v_2 = np.load(eigen_vec+"_1.npy").squeeze()
    with open(data+"/ref_pairs", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            for j in range(len(d[i])):
                d[i][j] = int(d[i][j])
            ent_dic_1[d[i][0]]=num_ent_1
            ent_dic_2[d[i][1]]=num_ent_2
            num_ent_1 += 1
            num_ent_2 += 1
        ref = len(d)
    with open(data+"/sup_pairs", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            for j in range(len(d[i])):
                d[i][j] = int(d[i][j])
            ent_dic_1[d[i][0]]=num_ent_1
            ent_dic_2[d[i][1]]=num_ent_2
            num_ent_1 += 1
            num_ent_2 += 1
        sup = len(d)

    with open(data+"/ent_ids_1", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            tmp = int(d[i][0])
            if tmp not in ent_dic_1:
                ent_dic_1[tmp] = num_ent_1
                num_ent_1 += 1

    with open(data+"/ent_ids_2", "r", encoding = "utf-8") as f:
        d = f.readlines()
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            tmp = int(d[i][0])
            if tmp not in ent_dic_2:
                ent_dic_2[tmp] = num_ent_2
                num_ent_2 += 1
    
    with open(data+"/triples_1", "r", encoding = "utf-8") as f:
        d = f.readlines()
        s_i = [[],[]]
        s_v = []
        eye_i = [[],[]]
        eye_v = []
        h_t = dict([x,dict()] for x in range(num_ent_1))
        
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            d[i][0] = ent_dic_1[int(d[i][0])]
            d[i][2] = ent_dic_1[int(d[i][2])]
            
            if d[i][0] == d[i][2]:
                continue
            if d[i][2] not in h_t[d[i][0]].keys():
                h_t[d[i][0]][d[i][2]] = 1
            else:
                h_t[d[i][0]][d[i][2]] += 1
            if d[i][0] not in h_t[d[i][2]].keys():
                h_t[d[i][2]][d[i][0]] = 1
            else:
                h_t[d[i][2]][d[i][0]] += 1
    
        # sparse adj for entity&entity:
        # undirect single graph
        for i in range(num_ent_1):
            t_ev = list(h_t[i].keys())
            t_ev.append(i)
            ev = beta*e_v_1[t_ev]/e_v_1[i]
            ev = torch.softmax(torch.tensor(ev),dim=0).numpy()
            for j in range(len(t_ev)):
                s_i[0].append(i)
                s_i[1].append(t_ev[j])
                s_v.append(ev[j])
            eye_i[0].append(i)
            eye_i[1].append(i)
            eye_v.append(ev[-1])
        eye_i = torch.LongTensor(eye_i)
        eye_v = torch.FloatTensor(eye_v)
        S_eye = torch.sparse.FloatTensor(eye_i, eye_v, torch.Size((num_ent_1, num_ent_1))).cuda()

        s_i = torch.LongTensor(s_i)
        s_v = torch.FloatTensor(s_v)
        S_adj_1 = torch.sparse.FloatTensor(s_i, s_v, torch.Size((num_ent_1, num_ent_1))).cuda()
        h_t1 = h_t

    with open(data+"/triples_2", "r", encoding = "utf-8") as f:
        d = f.readlines()
        s_i = [[],[]]
        s_v = []
        eye_i = [[],[]]
        eye_v = []
        h_t = dict([x,dict()] for x in range(num_ent_2))
        
        for i in range(len(d)):
            d[i] = d[i].strip("\n").split("\t")
            d[i][0] = ent_dic_2[int(d[i][0])]
            d[i][2] = ent_dic_2[int(d[i][2])]
            
            if d[i][0] == d[i][2]:
                continue
            if d[i][2] not in h_t[d[i][0]].keys():
                h_t[d[i][0]][d[i][2]] = 1
            else:
                h_t[d[i][0]][d[i][2]] += 1
            if d[i][0] not in h_t[d[i][2]].keys():
                h_t[d[i][2]][d[i][0]] = 1
            else:
                h_t[d[i][2]][d[i][0]] += 1
        
        # sparse adj for entity&entity:
        # undirect single graph
        for i in range(num_ent_2):
            t_ev = list(h_t[i].keys())
            t_ev.append(i)
            ev = beta*e_v_2[t_ev]/e_v_2[i]
            ev = torch.softmax(torch.tensor(ev),dim=0).numpy()
            for j in range(len(t_ev)):
                s_i[0].append(i)
                s_i[1].append(t_ev[j])
                s_v.append(ev[j])
            eye_i[0].append(i)
            eye_i[1].append(i)
            eye_v.append(ev[-1])
        eye_i = torch.LongTensor(eye_i)
        eye_v = torch.FloatTensor(eye_v)
        T_eye = torch.sparse.FloatTensor(eye_i, eye_v, torch.Size((num_ent_2, num_ent_2))).cuda()
        
        s_i = torch.LongTensor(s_i)
        s_v = torch.FloatTensor(s_v)
        S_adj_2 = torch.sparse.FloatTensor(s_i, s_v, torch.Size((num_ent_2,num_ent_2))).cuda()
        h_t2 = h_t
    del s_i, s_v, h_t, d
    return ref, sup, num_ent_1, num_ent_2, S_adj_1, S_adj_2, h_t1, h_t2, S_eye, T_eye, ent_dic_1, ent_dic_2



def loadfile(fn, num=1):
	print('loading a file...' + fn)
	ret = []
	with open(fn, encoding='utf-8') as f:
		for line in f:
			th = line[:-1].split('\t')
			x = []
			for i in range(num):
				x.append(int(th[i]))
			ret.append(tuple(x))
	return ret

#def rfunc(KG, e):
def rfunc(KG):
    head = {} #key是关系和实体的组合，value是头实体的集合
    cnt = {} #key是关系和实体的组合，value是相关实体的数量（
    rel_type = {}
    cnt_r = {}
    for tri in KG:
        r_e = str(tri[1]) + ' ' + str(tri[2])
        if r_e not in cnt:
            cnt[r_e] = 1
            head[r_e] = set([tri[0]])
        else:
            cnt[r_e] += 1
            head[r_e].add(tri[0])
        if tri[1] not in cnt_r:
            cnt_r[tri[1]] = 1
    r_num = len(cnt_r)
    for r_e in cnt:
        value = 1.0 * len(head[r_e]) / cnt[r_e]
        rel_type[r_e] = value
    del cnt
    del head
    del cnt_r
    cnt = {}
    head = {}
    for tri in KG:
        r_e_new = str(tri[1]+r_num) + ' ' + str(tri[0])
        if r_e_new not in cnt:
            cnt[r_e_new] = 1
            head[r_e_new] = set([tri[2]])
        else:
            cnt[r_e_new] += 1
            head[r_e_new].add(tri[2])
    for r_e in cnt:
        value = 1.0 * len(head[r_e]) / cnt[r_e]
        rel_type[r_e] = value
    return rel_type, r_num
    '''
    #e1 = set(loadfile(config.e1, 1))
    #e2 = set(loadfile(config.e2, 1))
    #e = len(e1 | e2)
    #e是输入，来源如上；没有从KG直接读取，可能是考虑有孤立实体？
    head_r = np.zeros((e, r_num))
    tail_r = np.zeros((e, r_num))
    for tri in KG:
        head_r[tri[0]][tri[1]] = 1
        tail_r[tri[2]][tri[1]] = 1
    return head_r, tail_r, rel_type
    '''
    

def load_rnm_data(data):
    #data = 'data/dbp15k/zh_en/mtranse/0_3'
    rdata = {}
    rdata['train']=loadfile(data + '/sup_pairs',2)
    rdata['test_pair'] =loadfile(data + '/ref_pairs',2)
    KG1 = loadfile(data + '/triples_1', 3)
    KG2 = loadfile(data + '/triples_2', 3)
    rdata['KG'] = KG1 + KG2
    rdata['rel_type'], rdata['r_num'] = rfunc(rdata['KG'])
    #train=loadfile(config.traindir,2)
    #KG1 = loadfile(config.kg1, 3)
    #KG2 = loadfile(config.kg2, 3)
    #KG = KG1 + KG2
    #_, _, rel_type = rfunc(KG, e)
    #train, KG1 + KG2, rel_type
    
    return rdata
