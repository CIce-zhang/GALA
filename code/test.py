import gc
import multiprocessing
import time

import numpy as np
from scipy.spatial.distance import cdist
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from code.utils import act_f



def stable_test(model,op,epo, S_in, T_in,S_ad, T_ad, s_mem, t_mem, Test_list, maxa, maxe, file, a_test):
    model.eval()
    te_time = time.time()
    emb = model.state_dict()["Eemb.weight"]
    se = emb[S_in]
    te = emb[T_in]
    s = model.L1(se)
    t = model.L1(te)
    s = S_ad.mm(s)
    t = T_ad.mm(t)
    if op["gcn_num"] > 1:
        #more layer remain to be optimized
        s2 = S_ad.mm(s)
        t2 = T_ad.mm(t)
        s = torch.cat((s,s2),-1)
        t = torch.cat((t,t2),-1)
        del s2,t2
    acc=0
    sim_mat, hits, ran, rran = Soft_align(s,t,Test_list,op["h@k"],op["test_metric"],op["csls_k"])
    print("-"*30)
    print("Hit@{} : {} ({:.3%}) MR : {:.3f} MMR: {:.3f} | time: {:.3f}".format(op["h@k"], hits, hits[0]/op["align_num"], ran/op["align_num"], rran/op["align_num"], time.time()-te_time))
    m_s, count = Stable_align(sim_mat, Test_list, op["max_iteration"], op["accurate"])
    if maxa < count:
        maxa = count
        maxe = epo
    print("Current best: Epoch {}, Hits {}".format(maxe,maxa))
    tag = torch.arange(op["align_num"])
    s_weight = sim_weight(op,sim_mat)
    t_weight = sim_weight(op,sim_mat.t())
    sim_mat = op["soft_beta"]*sim_mat
    s_out = torch.mm(torch.softmax(sim_mat,dim=1),te)
    t_out = torch.mm(torch.softmax(sim_mat.t(),dim=1),se)
    if epo == 0:
        s_mem = s_out
        t_mem = t_out
    else:
        s_mem = torch.add((1-op["soft_mem"])*s_mem, op["soft_mem"]*s_out)
        t_mem = torch.add((1-op["soft_mem"])*t_mem, op["soft_mem"]*t_out)
    s_out = torch.add(op["soft_alpha"]*se, (1-op["soft_alpha"])*s_mem)
    t_out = torch.add(op["soft_alpha"]*te, (1-op["soft_alpha"])*t_mem)
    s_out[op["align_num"]:op["align_num"]+op["sup_num"]] = (s_out[op["align_num"]:op["align_num"]+op["sup_num"]] + t_out[op["align_num"]:op["align_num"]+op["sup_num"]])/2
    t_out[op["align_num"]:op["align_num"]+op["sup_num"]] = s_out[op["align_num"]:op["align_num"] + op["sup_num"]]
    
    del emb,s,t,se,te,sim_mat, m_s
    #model.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()
    return maxa, maxe, s_out, t_out, s_mem, t_mem, s_weight, t_weight

def Stable_align(sim_mat,test_list,max_iteration, accurate):
    print("-"*30)
    print("SM: Start processing")
    #sim_mat = sim_mat.cpu()
    ct = time.time()
    print(sim_mat.shape)
    #r = torch.topk(sim_mat,sim_mat.shape[0],dim=1,largest=True,sorted=True)
    r = torch.topk(sim_mat,sim_mat.shape[1],dim=1,largest=True,sorted=True)
    rank = r[1].tolist()
    d1=dict([i,rank[i]] for i in range(len(rank)))
    #r = torch.topk(sim_mat.t(),sim_mat.shape[0],dim=1,largest=True,sorted=True)
    r = torch.topk(sim_mat.t(),sim_mat.shape[1],dim=1,largest=True,sorted=True)
    rank = r[1].tolist()
    d2=dict([i,rank[i]] for i in range(len(rank)))
    print("SM: Processing takes %2.4f"%(time.time()-ct))
    ct = time.time()
    print("SM: Start matching")
    m_s = galeshapley(d1, d2, max_iteration, accurate)
    count = 0
    for i in test_list:
        if i in m_s.keys() and m_s[i] == i:
            count += 1
    print("SM: Matching results: {}({:.3%}) , matching rate: {}({:.3%}), takes {:.3}s".format(count, float(count)/len(test_list), len(m_s.keys()), float(len(m_s.keys()))/len(test_list), time.time()-ct))
    print("-"*30)
    del r, rank, d1, d2
    return m_s, count

def galeshapley(suitor_pref_dict, reviewer_pref_dict, max_iteration, accurate):
    """ The Gale-Shapley algorithm. This is known to provide a unique, stable
    suitor-optimal matching. The algorithm is as follows:
    (1) Assign all suitors and reviewers to be unmatched.
    (2) Take any unmatched suitor, s, and their most preferred reviewer, r.
            - If r is unmatched, match s to r.
            - Else, if r is matched, consider their current partner, r_partner.
                - If r prefers s to r_partner, unmatch r_partner from r and
                  match s to r.
                - Else, leave s unmatched and remove r from their preference
                  list.
    (3) Go to (2) until all suitors are matched, then end.
    Parameters
    ----------
    suitor_pref_dict : dict
        A dictionary with suitors as keys and their respective preference lists
        as values
    reviewer_pref_dict : dict
        A dictionary with reviewers as keys and their respective preference
        lists as values
    max_iteration : int
        An integer as the maximum iterations
    Returns
    -------
    matching : dict
        The suitor-optimal (stable) matching with suitors as keys and the
        reviewer they are matched with as values
    """
    suitors = list(suitor_pref_dict.keys())
    matching = dict()
    rev_matching = dict()

    for i in range(max_iteration):
        if len(suitors) <= 0:
            break
        for s in suitors:
            r = suitor_pref_dict[s][0]
            if r not in matching.values():
                matching[s] = r
                rev_matching[r] = s
            else:
                r_partner = rev_matching.get(r)
                if reviewer_pref_dict[r].index(s) < reviewer_pref_dict[r].index(r_partner):
                    del matching[r_partner]
                    matching[s] = r
                    rev_matching[r] = s
                else:
                    suitor_pref_dict[s].remove(r)
        suitors = list(set(suitor_pref_dict.keys()) - set(matching.keys()))
    if accurate=="T" :
        for s in suitors:
            matching[s] = suitor_pref_dict[s][0]
    return matching

def test_align(embed1, embed2, test_list, metric, csls_k):
    sim_mat = Cuda_sim(embed1, embed2, metric=metric, csls_k=csls_k)
    #sim_mat = torch.from_numpy(sim_mat).cuda()
    rank = torch.max(sim_mat, dim=1)
    print(rank[1][test_list].shape)
    hit = torch.eq(rank[1][test_list],torch.tensor(test_list).cuda())
    hit_list = list(zip(test_list,rank[1][test_list].tolist()))
    ct = 0
    for p in hit_list:
        if p[0]==p[1]:
            ct+=1
    assert (torch.sum(hit) - ct) == 0
    #sim_mat = sim_mat.to_dense()
    #deg = [len(torch.nonzero(sim_mat[test_list][i])) for i in range(sim_mat[test_list].shape[0])]
    del sim_mat
    gc.collect()
    torch.cuda.empty_cache()
    return hit,hit_list

def deg_test(model,op,epo, S_in, T_in,S_ad, T_ad, s_mem, t_mem, Test_list, maxa, maxe, file, a_test):
    model.eval()
    te_time = time.time()
    emb = model.state_dict()["Eemb.weight"]
    se = emb[S_in]
    te = emb[T_in]
    s = model.L1(se)
    t = model.L1(te)
    s = S_ad.mm(s)
    t = T_ad.mm(t)
    if op["gcn_num"] > 1:
        #more layer remain to be optimized
        s2 = S_ad.mm(s)
        t2 = T_ad.mm(t)
        s = torch.cat((s,s2),-1)
        t = torch.cat((t,t2),-1)
        del s2,t2
    acc=0
    hit,result = test_align(s,t,Test_list,op["test_metric"],op["csls_k"])
    '''if maxa < hit:
        maxa = hit
        maxe = epo'''
    print("-"*30)
    print("Hit@1 : {} ({:.3%})".format(torch.sum(hit), torch.sum(hit).float()/op["align_num"]))
    #print("Hit@{} : {} ({:.3%}) MR : {:.3f} MMR: {:.3f} | max: Epoch {:d} Acc {:d}".format(op["h@k"], hits, hits[0]/op["align_num"], ran/op["align_num"], rran/op["align_num"], maxe, maxa))
    del emb,s,t,se,te
    #model.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()
    return maxa, maxe, hit, result

def test_test(model,op,epo, S_in, T_in,S_ad, T_ad, s_mem, t_mem, Test_list, maxa, maxe, file, a_test):
    model.eval()
    te_time = time.time()
    emb = model.state_dict()["Eemb.weight"]
    se = emb[S_in]
    te = emb[T_in]
    #print(torch.sum(torch.eq(se[op["align_num"]:op["align_num"] + op["sup_num"]],te[op["align_num"]:op["align_num"] + op["sup_num"]])))
    #TKDE change
    print("-"*30)
    if op['reduce_test'] == 'T':
        sim_mat, hits, ran, rran = Soft_align(se[Test_list],te[Test_list],Test_list,op["h@k"],op["test_metric"],op["csls_k"])
        #print(sim_mat.size())
        get_hits_ma(sim_mat.cpu().numpy(),Test_list)
    
    sim_mat, hits, ran, rran = Soft_align(se,te,Test_list,op["h@k"],op["test_metric"],op["csls_k"])
    print("Input Layer: Hit@{} : {} ({:.3%}) MR : {:.3f} MMR: {:.3f} | max: Epoch {:d} Acc {:d}".format(op["h@k"], hits, hits[0]/op["align_num"], ran/op["align_num"], rran/op["align_num"], maxe, maxa))
    del hits,ran,rran
    
    s = model.L1(se)
    t = model.L1(te)
    #TKDE change
    print("-"*30)
    if op['reduce_test'] == 'T':
        sim_mat, hits, ran, rran = Soft_align(s[Test_list],t[Test_list],Test_list,op["h@k"],op["test_metric"],op["csls_k"])
        #print(sim_mat.size())
        get_hits_ma(sim_mat.cpu().numpy(),Test_list)
    
    sim_mat, hits, ran, rran = Soft_align(s,t,Test_list,op["h@k"],op["test_metric"],op["csls_k"])
    print("Linear Layer: Hit@{} : {} ({:.3%}) MR : {:.3f} MMR: {:.3f} | max: Epoch {:d} Acc {:d}".format(op["h@k"], hits, hits[0]/op["align_num"], ran/op["align_num"], rran/op["align_num"], maxe, maxa))
    del hits,ran,rran

    s = S_ad.mm(s)
    t = T_ad.mm(t)
    #TKDE change
    print("-"*30)
    if op['reduce_test'] == 'T':
        sim_mat, hits, ran, rran = Soft_align(s[Test_list],t[Test_list],Test_list,op["h@k"],op["test_metric"],op["csls_k"])
        #print(sim_mat.size())
        get_hits_ma(sim_mat.cpu().numpy(),Test_list)
    
    sim_mat, hits, ran, rran = Soft_align(s,t,Test_list,op["h@k"],op["test_metric"],op["csls_k"])
    print("GCN 1 Layer: Hit@{} : {} ({:.3%}) MR : {:.3f} MMR: {:.3f} | max: Epoch {:d} Acc {:d}".format(op["h@k"], hits, hits[0]/op["align_num"], ran/op["align_num"], rran/op["align_num"], maxe, maxa))
    del hits,ran,rran
    if op["gcn_num"] > 1:
        #more layer remain to be optimized
        s2 = S_ad.mm(s)
        t2 = T_ad.mm(t)
        #TKDE change
        print("-"*30)
        if op['reduce_test'] == 'T':
            sim_mat, hits, ran, rran = Soft_align(s2[Test_list],t2[Test_list],Test_list,op["h@k"],op["test_metric"],op["csls_k"])
            #print(sim_mat.size())
            get_hits_ma(sim_mat.cpu().numpy(),Test_list)
        
        sim_mat, hits, ran, rran = Soft_align(s2,t2,Test_list,op["h@k"],op["test_metric"],op["csls_k"])
        print("GCN 2 Layer: Hit@{} : {} ({:.3%}) MR : {:.3f} MMR: {:.3f} | max: Epoch {:d} Acc {:d}".format(op["h@k"], hits, hits[0]/op["align_num"], ran/op["align_num"], rran/op["align_num"], maxe, maxa))
        del hits,ran,rran
        s = torch.cat((s,s2),-1)
        t = torch.cat((t,t2),-1)
        del s2,t2
        #TKDE change
        print("-"*30)
        if op['reduce_test'] == 'T':
            sim_mat, hits, ran, rran = Soft_align(s[Test_list],t[Test_list],Test_list,op["h@k"],op["test_metric"],op["csls_k"])
            #print(sim_mat.size())
            get_hits_ma(sim_mat.cpu().numpy(),Test_list)
        
        sim_mat, hits, ran, rran = Soft_align(s,t,Test_list,op["h@k"],op["test_metric"],op["csls_k"])
        print("GCN Conc Layer: Hit@{} : {} ({:.3%}) MR : {:.3f} MMR: {:.3f} | max: Epoch {:d} Acc {:d}".format(op["h@k"], hits, hits[0]/op["align_num"], ran/op["align_num"], rran/op["align_num"], maxe, maxa))
        del hits,ran,rran
    acc=0
    #TKDE change
    print("-"*30)
    if op['reduce_test'] == 'T':
        sim_mat, hits, ran, rran = Soft_align(s[Test_list],t[Test_list],Test_list,op["h@k"],op["test_metric"],op["csls_k"])
        #print(sim_mat.size())
        get_hits_ma(sim_mat.cpu().numpy(),Test_list)
    
    sim_mat, hits, ran, rran = Soft_align(s,t,Test_list,op["h@k"],op["test_metric"],op["csls_k"])
    if maxa < hits[0]:
        maxa = hits[0]
        maxe = epo
    #TKDE change
    #_, hits, ran, rran = Soft_align(s,t,Test_list,op["h@k"],op["test_metric"],op["csls_k"])
    #print("-"*30)
    print("Hit@{} : {} ({:.3%}) MR : {:.3f} MMR: {:.3f} | max: Epoch {:d} Acc {:d}".format(op["h@k"], hits, hits[0]/op["align_num"], ran/op["align_num"], rran/op["align_num"], maxe, maxa))
    #del emb,s,t,se,te,sim_mat
    #model.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()
    #return maxa, maxe, s_out, t_out, s_mem, t_mem, s_weight, t_weight
    return maxa, maxe


def Soft_bt_test(model,op,epo, S_in, T_in,S_ad, T_ad, s_mem, t_mem, Test_list, maxa, maxe, file, a_test):
    model.eval()
    te_time = time.time()
    emb = model.state_dict()["Eemb.weight"]
    se = emb[S_in]
    te = emb[T_in]
    print("s_emb std: %1.7f mean: %1.7f"%(torch.std(se),torch.mean(se)))
    print("t_emb std: %1.7f mean: %1.7f"%(torch.std(te),torch.mean(te)))
    #TKDE change
    print("Test type:",op['test_emb'])
    if op['test_emb'] == 'input':
        s = se
        t = te
    elif op['test_emb'] == 'raw':
        s = model.L1(se)
        t = model.L1(te)
        s = S_ad.mm(s)
        t = T_ad.mm(t)
        if op["gcn_num"] > 1:
            #more layer remain to be optimized
            s2 = S_ad.mm(s)
            t2 = T_ad.mm(t)
            s = torch.cat((s,s2),-1)
            t = torch.cat((t,t2),-1)
            del s2,t2
    acc=0
    print("s_final_emb std: %1.7f mean: %1.7f"%(torch.std(s),torch.mean(s)))
    print("t_final_emb std: %1.7f mean: %1.7f"%(torch.std(t),torch.mean(t)))
    #TKDE change
    sim_mat, hits, ran, rran = Soft_align(s,t,Test_list,op["h@k"],op["test_metric"],op["csls_k"])
    #'''
    if op['test_emb'] == 'input' : #and op["test_metric"]=='cityblock'
        print("Notice"*10)
        s2 = model.L1(se)
        t2 = model.L1(te)
        s2 = S_ad.mm(s2)
        t2 = T_ad.mm(t2)
        if op["gcn_num"] > 1:
            #more layer remain to be optimized
            s3 = S_ad.mm(s2)
            t3 = T_ad.mm(t2)
            s2 = torch.cat((s2,s3),-1)
            t2 = torch.cat((t2,t3),-1)
            del s3,t3
        sim_mat, _, _, _ = Soft_align(s2,t2,Test_list,op["h@k"],'euclidean',op["csls_k"])
    
    if op['reduce_test'] == 'T':
        print("Basic Test: Hit@{} : {} ({:.3%}) MR : {:.3f} MMR: {:.3f}".format(op["h@k"], hits, hits[0]/op["align_num"], ran/op["align_num"], rran/op["align_num"]))
        _, hits, ran, rran = Soft_align(s[Test_list],t[Test_list],Test_list,op["h@k"],op["test_metric"],op["csls_k"])
    tag = torch.arange(op["align_num"])
    s_weight = sim_weight(op,sim_mat)
    t_weight = sim_weight(op,sim_mat.t())
    sim_mat = op["soft_beta"]*sim_mat
    s_out = torch.mm(torch.softmax(sim_mat,dim=1),te)
    t_out = torch.mm(torch.softmax(sim_mat.t(),dim=1),se)
    if epo == 0:
        s_mem = s_out
        t_mem = t_out
    else:
        s_mem = torch.add((1-op["soft_mem"])*s_mem, op["soft_mem"]*s_out)
        t_mem = torch.add((1-op["soft_mem"])*t_mem, op["soft_mem"]*t_out)
    
    s_out = torch.add(op["soft_alpha"]*se, (1-op["soft_alpha"])*s_mem)
    t_out = torch.add(op["soft_alpha"]*te, (1-op["soft_alpha"])*t_mem)

    s_out[op["align_num"]:op["align_num"]+op["sup_num"]] = (s_out[op["align_num"]:op["align_num"]+op["sup_num"]] + t_out[op["align_num"]:op["align_num"]+op["sup_num"]])/2
    t_out[op["align_num"]:op["align_num"]+op["sup_num"]] = s_out[op["align_num"]:op["align_num"] + op["sup_num"]]
    print("Update s_emb std: %1.7f mean: %1.7f"%(torch.std(s_out),torch.mean(s_out)))
    print("Update t_emb std: %1.7f mean: %1.7f"%(torch.std(t_out),torch.mean(t_out)))
    if maxa < hits[0]:
        maxa = hits[0]
        maxe = epo
    
    print("-"*30)
    print("Hit@{} : {} ({:.3%}) MR : {:.3f} MMR: {:.3f} | max: Epoch {:d} Acc {:d}".format(op["h@k"], hits, hits[0]/op["align_num"], ran/op["align_num"], rran/op["align_num"], maxe, maxa))
    if op["softalign"] == "T":
        if op['test_emb'] == 'input':
            s = s_out
            t = t_out
        elif op['test_emb'] == 'raw':              
            s = model.L1(s_out)
            t = model.L1(t_out)
            s = S_ad.mm(s)
            t = T_ad.mm(t)
            if op["gcn_num"] > 1:
                #more layer remain to be optimized
                s2 = S_ad.mm(s)
                t2 = T_ad.mm(t)
                s = torch.cat((s,s2),-1)
                t = torch.cat((t,t2),-1)
                del s2,t2
        acc=0
        print("Update s_final_emb std: %1.7f mean: %1.7f"%(torch.std(s),torch.mean(s)))
        print("Update t_final_emb std: %1.7f mean: %1.7f"%(torch.std(t),torch.mean(t)))
        if op['reduce_test'] == 'T':
            _, hits, ran, rran = Soft_align(s[Test_list],t[Test_list],Test_list,op["h@k"],op["test_metric"],op["csls_k"])
        else:
            _, hits, ran, rran = Soft_align(s,t,Test_list,op["h@k"],op["test_metric"],op["csls_k"])
        print("Update Hit@{} : {} ({:.3%}) MR : {:.3f} MMR: {:.3f}".format(op["h@k"], hits, hits[0]/op["align_num"], ran/op["align_num"], rran/op["align_num"]))
        print("Test time: %4.1f"%(time.time()-te_time))
        print("-"*30)
    del emb,s,t,se,te,sim_mat
    #model.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()
    return maxa, maxe, s_out, t_out, s_mem, t_mem, s_weight, t_weight

def Soft_align(embed1, embed2, test_list, top_k, metric, csls_k):
    sim_mat = Cuda_sim(embed1, embed2, metric=metric, csls_k=csls_k)
    #sim_mat = torch.from_numpy(sim_mat).cuda()
    num = len(test_list)
    hits, _, _, mr_list = Cuda_calculate_rank(test_list, sim_mat, top_k, num)
    ra = torch.sum(mr_list)
    rra = torch.sum(torch.div(1,mr_list.float()))
    del mr_list
    gc.collect()
    torch.cuda.empty_cache()
    return sim_mat, hits, ra, rra

def sim_weight(op, sim_mat):
    #s_sim = torch.softmax(sim_mat,dim=1)
    sim = sim_mat
    m_sim = torch.max(sim_mat,dim=1)[0]
    print(m_sim)
    print("Anchot num: %d | Anchor mean %1.7f "%(torch.sum(torch.eq(torch.max(sim_mat,dim=1)[1][op["align_num"]:op["align_num"] + op["sup_num"]],torch.arange(op["align_num"],op["align_num"] + op["sup_num"]).cuda())),torch.mean(m_sim[op["align_num"]:op["align_num"] + op["sup_num"]])))
    re = torch.ge(m_sim,torch.tensor(op["zoom_trunc"]))
    print("Trunc num: %d | Trunc Acc: %d"%(torch.sum(re),torch.sum(re[:op["align_num"] + op["sup_num"]]*(torch.eq(torch.max(sim_mat,dim=1)[1][:op["align_num"] + op["sup_num"]],torch.arange(op["align_num"] + op["sup_num"]).cuda())))))
    up_bound = torch.max(m_sim)
    if op["zoom_type"] == "linear": 
        weight = (m_sim - op["zoom_trunc"])/(up_bound - op["zoom_trunc"])
        weight = re.float()*op["normalize"]*weight+1
    elif op["zoom_type"] == "power": 
        if op["zoom_base"] == 1:
            weight = torch.pow(math.e,(m_sim - op["zoom_trunc"])/(up_bound - op["zoom_trunc"])*math.log(op["normalize"]))
        else:
            weight = torch.pow(op["zoom_base"],(m_sim - op["zoom_trunc"])/(up_bound - op["zoom_trunc"])*math.log(op["normalize"],op["zoom_base"]))
        weight = re.float()*weight+1
    else:
        weight = torch.zeros_like(m_sim)
        weight = re.float()*op["normalize"]*weight+1
    
    weight[op["align_num"]:op["align_num"]+op["sup_num"]] = op["normalize"]
    print("Mean weight: %d"%torch.mean(weight))
    '''
    state = {"weight":weight, "sim":m_sim, "re":re}
    torch.save(state,"comp2")
    '''
    return weight


def Cuda_calculate_rank(idx, sim_mat, top_k, total_num):
    assert 1 in top_k
    mr = 0
    mrr = 0
    hits = [0] * len(top_k)
    hits1_rest = []
    '''
    # use topk for H@k,
    # use sum(V[pos]>V) for mr&mrr
    rank = torch.topk(sim_mat, top_k[-1], dim=1, sorted=True)
    '''
    rank = torch.topk(sim_mat, top_k[-1], dim=1, largest = True, sorted = True)
    for i in range(len(top_k)):
        hits[i] = torch.sum(torch.eq(rank[1][idx][:,:top_k[i]],torch.tensor(idx).cuda().unsqueeze(1))).item()
    mr = torch.sum(torch.ge(sim_mat[idx],sim_mat[[idx,idx]].unsqueeze(1)),dim=1)
    #mr /= total_num
    #mrr /= total_num
    return hits, rank[0][idx,0], rank[1][idx,0], mr

def Cuda_sim(embed1, embed2, metric='inner', csls_k=0):
    #TKDE change
    #sim_mat = -torch.cdist(embed1, embed2)
    if metric == 'euclidean':
        sim_mat = -torch.cdist(embed1, embed2,p=2)
    elif metric == 'cityblock':
        sim_mat = -torch.cdist(embed1, embed2,p=1)
    elif metric == 'inner':
        sim_mat = torch.matmul(F.normalize(embed1,p=2,dim=-1), F.normalize(embed2,p=2,dim=-1).t())
    
    if csls_k > 0:
        sim_mat = Cuda_csls_sim(sim_mat, csls_k)
    return sim_mat

def Cuda_csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.

    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.
    """
    nearest_values1 = torch.mean(torch.topk(sim_mat,k,dim=1)[0],dim=1, keepdim=True)
    nearest_values2 = torch.mean(torch.topk(sim_mat.t(),k,dim=1)[0],dim=1, keepdim=True)
    csls_sim_mat = 2 * sim_mat - nearest_values1 - nearest_values2.T
    return csls_sim_mat

def get_hits_ma(sim, test_pair, top_k=(1, 10)):
    top_lr = [0] * len(top_k)
    mrr_sum_l = 0
    for i in range(sim.shape[0]):
        #TKDE change
        #rank = sim[i, :].argsort()
        rank = sim[i, :].argsort()[::-1]
        # print(sim[i, :])
        rank_index = np.where(rank == i)[0][0]
        mrr_sum_l = mrr_sum_l + 1.0 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
    msg = 'Hits@1:%.3f, Hits@10:%.3f, MRR:%.3f' % (top_lr[0] / len(test_pair), top_lr[1] / len(test_pair), mrr_sum_l / len(test_pair))
    print(msg)



#vec+test_pair 可替换为l_vec+R_vec,
#l1可能需调整为1,或者vec_r变为全1的向量——好像也不太行，对应部分得删减先
#M0、rel_type/ref_data需原始读取数据获得，先loadfile、再build，该部分最好打包在一个函数当中

#def get_hits_v1(vec, vec_r, M0, ref_data, rel_type, test_pair, sim_e, top_k=(1, 10)):
#rnm_test(Lvec ,Rvec , r_num, KG, train, rel_type, test, None)
#get_hits(outvec, outvec_r, KG, ILL, rel_type, test, None, None)

def rnm_test(model, op, epo, S_in, T_in, Test_list, maxa, maxe, rdata):
    model.eval()
    te_time = time.time()
    macc = -1.0
    lmacc = -1.0
    rmacc = -1.0
    emb = model.state_dict()["Eemb.weight"]
    se = emb[S_in]
    te = emb[T_in]
    s = se[Test_list].cpu().numpy()
    t = te[Test_list].cpu().numpy()
    assert len(Test_list) == len(rdata['test_pair'])
    for e1,e2 in rdata['test_pair']:
        if e1 != Test_list[e1] or e2!=e1+10500:
            print("Wrong Mapping!!",e1,e2)
    print("*"*30)
    print("*"*30)
    print("RNM testing...")
    for m in ['euclidean','cityblock']:
        msg = m + " Metric itreation"
        print(msg,' iter 1')
        sim_e, lm, rm = get_hits_rnm(s, t, rdata['r_num'], rdata['KG'], rdata['train'], rdata['rel_type'], rdata['test_pair'], m, None)
        lmacc = max(lm,lmacc)
        rmacc = max(rm,rmacc)
        macc = max(macc, lm, rm)
        for j in range(3):
            print(msg,' iter ',j+2)
            sim_e, lm, rm  = get_hits_rnm(s, t, rdata['r_num'], rdata['KG'], rdata['train'], rdata['rel_type'], rdata['test_pair'], m, sim_e)
            lmacc = max(lm,lmacc)
            rmacc = max(rm,rmacc)
            macc = max(macc, lm, rm)
    if macc > maxa:
        maxa = macc
        maxe = epo
    print("Best Epoch: %d, %.2f%% ; Current Epoch left max: %.2f%%, right max: %.2f%%"%(maxe,maxa,lmacc,rmacc))
    print("Test time: %4.1f"%(time.time()-te_time))
    print("*"*30)
    print("*"*30)
    #get_hits_rnm 可用于替换原来的soft_align函数
    #该部分尝试的可以的话，可以继承至Soft Align过程中
    return maxa,maxe


def get_hits_rnm(Lvec, Rvec, r_num, M0, ref_data, rel_type, test_pair, met, sim_e, top_k=(1, 10)):
    ref = set()
    for pair in ref_data:
        ref.add((pair[0], pair[1]))
    
    #r_num = len(vec_r)//2
    #r_num相当于是关系的总个数
    #vec_r的长度是关系总数的两倍，因为每个关系和其逆关系向量是同时保存的，具体如下：
    '''
    #Model.py
    r_forward=tf.concat([L,R],axis=-1)
    r_reverse=tf.concat([-L,-R],axis=-1)
    r_embeddings = tf.concat([r_forward,r_reverse], axis=0)
    '''
    
    #kg/rel_ent用于提取论文中的T1、T2，M0是整个KG；要想在GALA中引入，需注意该处是一次性输入了两个KG
    kg = {}
    rel_ent = {}
    for tri in M0:
        if tri[0] == tri[2]:
            continue
        if tri[0] not in kg:
            kg[tri[0]] = set()
        if tri[2] not in kg:
            kg[tri[2]] = set()
        if tri[1] not in rel_ent:
            rel_ent[tri[1]] = set()
        
        kg[tri[0]].add((tri[1], tri[2]))
        kg[tri[2]].add((tri[1]+r_num, tri[0])) # tri[1]+r_num相当于对某个关系的id，转换为其逆关系的id
        rel_ent[tri[1]].add((tri[0], tri[2]))

    
    L = np.array([e1 for e1, e2 in test_pair])
    R = np.array([e2 for e1, e2 in test_pair])
    #Lvec = vec[L]
    #Rvec = vec[R]
    
    sim = cdist(Lvec, Rvec, metric=met)
    
    
    #迭代扩充seed
    #sim_e是上一轮(第一轮的话则是本轮)的相似度对齐结果
    #ref是对齐种子合集。此前已经读入了golden pair，此处对应迭代策略，试图引入新的set
    #R_set用于暂存推理的对齐结果
    if sim_e is None:
        sim_e = sim
    
    R_set = {}
    for i in range(len(L)):
        j = sim_e[i, :].argsort()[0]
        if sim_e[i,j] >= 5:
            continue
        if j in R_set and sim_e[i, j] < R_set[j][1]:
            ref.remove((L[R_set[j][0]], R[j]))
            ref.add((L[i], R[j]))
            R_set[j] = (i, sim_e[i, j])
        if j not in R_set:
            ref.add((L[i], R[j]))
            R_set[j] = (i, sim_e[i, j])
    
    #rel_type是在构建KG的时候就读取了的
    #此处是唯一利用了建立的kg组合的地方
    for i in range(len(L)):
        rank = sim[i, :].argsort()[:100]
        for j in rank:
            if R[j] in kg: #不确定为何要if，应该必然是在KG当中的；补了个else输出看看; R[j]将目标实体的序号j转为了实体id
                match_num = 0
                for n_1 in kg[L[i]]: #对于源实体L[i]，找其所有相邻的n_1
                    for n_2 in kg[R[j]]: #遍历目标实体相邻的n_2
                        #if (n_1[1], n_2[1]) in ref and (n_1[0], n_2[0]) in ref_r: #对比n1、n2，看是否实体、关系均匹配上
                        if (n_1[1], n_2[1]) in ref : #change ,不再考虑关系是否匹配上
                            w = rel_type[str(n_1[0]) + ' ' + str(n_1[1])] * rel_type[str(n_2[0]) + ' ' + str(n_2[1])] #P(r1,n1)*P(r2,n2)
                            match_num += w
                sim[i,j] -= 10 * match_num / (len(kg[L[i]]) + len(kg[R[j]]))
            #else:
            #    print("Warnging! ",i,R[j]," not in KG")
    
    mrr_l = []
    mrr_r = []
    
    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        mrr_l.append(1.0 / (rank_index+1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        mrr_r.append(1.0 / (rank_index+1))
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
    
    left_max = -1
    right_max = -1
    print('Entity Alignment (left):')
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    left_max = max(100.0 * top_lr[0] / len(test_pair),left_max)
    print('MRR: %.4f' % (np.mean(mrr_l)))
    
    print('Entity Alignment (right):')
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    right_max = max(100.0 * top_rl[0] / len(test_pair),right_max)
    print('MRR: %.4f' % (np.mean(mrr_r)))
    
    return sim, left_max, right_max


