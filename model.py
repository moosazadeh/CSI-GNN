import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GNN
from torch.nn import Module, Parameter, Linear
import torch.nn.functional as F

# Combined Side Information-driven Graph Neural Networks
class CSI_GNN(Module):      
    def __init__(self, opt, num_item, num_brnd, num_cat, num_total, brand, category):
        super(CSI_GNN, self).__init__()
        self.opt = opt
        self.dataset = opt.dataset
        self.batch_size = opt.batch_size
        self.dim = opt.hiddenSize
        self.dropout_local = opt.dropout_local
        self.dropout_global = opt.dropout_global
        self.num_layer = opt.num_layer

        self.num_item = num_item
        self.num_brnd = num_brnd
        self.num_cat = num_cat
        self.num_total = num_total
        self.brand = brand
        self.category = category
        
        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)
        self.gnn = GNN(self.dim)
        self.local_agg_node = LocalAggregator(self.dim, self.opt.alpha, dropout=0.0)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_total, self.dim)
        self.pos_embedding = nn.Embedding(300, self.dim)

        # Parameters
        self.a1 = Parameter(torch.Tensor(1))
        self.a2 = Parameter(torch.Tensor(1))
        self.a3 = Parameter(torch.Tensor(1))
        self.a4 = Parameter(torch.Tensor(1))

        self.w_1 = nn.Parameter(torch.Tensor(4 * self.dim, 3*self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(3*self.dim, 1))
        self.glu1 = nn.Linear(3*self.dim, 3*self.dim)
        self.glu2 = nn.Linear(3*self.dim, 3*self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()
        
        all_brnds = []
        for x in range(1, num_item):
            all_brnds += [brand.get(x)]
        all_brnds = np.asarray(all_brnds)  
        self.all_brnds =  trans_to_cuda(torch.Tensor(all_brnds).long())
        
        all_cats= []
        for x in range(1, num_item):
            all_cats += [category.get(x)]
        all_cats = np.asarray(all_cats)  
        self.all_cats = trans_to_cuda(torch.Tensor(all_cats).long())

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, locl_itm_emb, local_brnd_based_itm_emb, local_cat_based_itm_emb,
                                                  locl_brnd_emb, locl_cat_emb, locl_brnd_itm_emb, locl_cat_itm_emb,
                                                                                  locl_itm_brnd_emb, locl_itm_cat_emb, mask):
        ''' Compute predicted probability tensor (scores) of all items in dataset. '''
        itm_emb = locl_itm_emb + local_brnd_based_itm_emb + local_cat_based_itm_emb + locl_brnd_itm_emb * self.a1 + locl_cat_itm_emb * self.a2
        brnd_emb = locl_brnd_emb + locl_itm_brnd_emb * self.a3
        cat_emb = locl_cat_emb + locl_itm_cat_emb * self.a4
        coll_itm_emb = torch.cat([itm_emb, brnd_emb, cat_emb],-1)     # Collaborative item embedding
        
        mask = mask.float().unsqueeze(-1)
        batch_size = itm_emb.shape[0]
        len = itm_emb.shape[1]
        
        pos_emb = (self.pos_embedding.weight[:len]).unsqueeze(0).repeat(batch_size, 1, 1)   # Position embedding
        z = torch.tanh(torch.matmul(torch.cat([pos_emb, coll_itm_emb], -1), self.w_1))
        
        # Collaborative session embedding
        coll_sess_emb = (torch.sum(coll_itm_emb * mask, -2) / torch.sum(mask, 1)).unsqueeze(-2).repeat(1, len, 1)

        # Attention coefficients
        beta = (torch.matmul(torch.sigmoid(self.glu1(z) + self.glu2(coll_sess_emb)), self.w_2))* mask
        coll_att_sess_emb = torch.sum(beta * coll_itm_emb, 1)   # Collaborative attention-based session embedding

        # Concatenate the embedding of each item and its corresponding brand and category.
        all_itm_emb = self.embedding.weight[1: self.num_item]
        all_brnd_emb = self.embedding(self.all_brnds)
        all_cat_emb = self.embedding(self.all_cats)
        all_itm_brnd_cat_emb = torch.cat([all_itm_emb, all_brnd_emb, all_cat_emb],-1)
	
	    # Calculate the prediction score of each candidate item
        scores = torch.matmul(coll_att_sess_emb, all_itm_brnd_cat_emb.transpose(1, 0))
        return scores

    def forward(self, sess_idxs, mask, sess_itms, usess_itms, local_adj_itms, 
                ubrnd_based_itms, local_adj_brnd_based_itms, ucat_based_itms, local_adj_cat_based_itms, 
                                                        usess_brnds, local_adj_brnds, usess_cats, local_adj_cats,
                                                         usess_itm_brnd, local_adj_itm_brnd, usess_itm_cat, local_adj_itm_cat):
       
        # Local Embedding
        local_itm_emb = self.embedding(usess_itms)
        local_brnd_based_itm_emb = self.embedding(ubrnd_based_itms)
        local_cat_based_itm_emb = self.embedding(ucat_based_itms)
        local_brnd_emb = self.embedding(usess_brnds)
        local_cat_emb = self.embedding(usess_cats)
        local_itm_brnd_emb = self.embedding(usess_itm_brnd)
        local_itm_cat_emb = self.embedding(usess_itm_cat)
        
        # Local aggregation
        local_itm_emb = self.local_agg(local_itm_emb, local_adj_itms, mask)
        local_brnd_based_itm_emb = self.local_agg(local_brnd_based_itm_emb, local_adj_brnd_based_itms, mask)
        local_cat_based_itm_emb = self.local_agg(local_cat_based_itm_emb, local_adj_cat_based_itms, mask)
        local_brnd_emb = self.gnn(local_adj_brnds, local_brnd_emb)
        local_cat_emb = self.gnn(local_adj_cats, local_cat_emb)
        local_itm_brnd_emb = self.local_agg_node(local_itm_brnd_emb, local_adj_itm_brnd, mask)
        local_itm_cat_emb = self.local_agg_node(local_itm_cat_emb, local_adj_itm_cat, mask)

            
       # Dropout
        local_itm_emb = F.dropout(local_itm_emb, self.dropout_local, training=self.training)
        local_brnd_based_itm_emb = F.dropout(local_brnd_based_itm_emb, self.dropout_local, training=self.training)
        local_cat_based_itm_emb = F.dropout(local_cat_based_itm_emb, self.dropout_local, training=self.training)
        local_brnd_emb = F.dropout(local_brnd_emb, self.dropout_local, training=self.training)
        local_cat_emb = F.dropout(local_cat_emb, self.dropout_local, training=self.training)
        local_itm_brnd_emb = F.dropout(local_itm_brnd_emb, self.dropout_local, training=self.training)
        local_itm_cat_emb = F.dropout(local_itm_cat_emb, self.dropout_local, training=self.training)

        return local_itm_emb, local_brnd_based_itm_emb, local_cat_based_itm_emb, local_brnd_emb, local_cat_emb, local_itm_brnd_emb, local_itm_cat_emb


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, bch_data):
            
    (sess_idxs, mask, targets, sess_itms,
     alias_sess_itms, local_adj_itms, usess_itms,
     alias_brnd_based_itms, local_adj_brnd_based_itms, ubrnd_based_itms,
     alias_cat_based_itms, local_adj_cat_based_itms, ucat_based_itms,
     alias_sess_brnds, local_adj_brnds, usess_brnds, 
     alias_sess_cats, local_adj_cats, usess_cats, 
     alias_brnd_itms, alias_itm_brnds, local_adj_itm_brnd, usess_itm_brnd,
     alias_cat_itms, alias_itm_cats, local_adj_itm_cat, usess_itm_cat)= bch_data
        
    sess_idxs = sess_idxs.numpy()
    mask = trans_to_cuda(mask).long()
    sess_itms = trans_to_cuda(sess_itms).long()
    
    # alias_sess_itms represents the relative position of the items clicked sequentially in each session.
    alias_sess_itms = trans_to_cuda(alias_sess_itms).long()
    local_adj_itms = trans_to_cuda(local_adj_itms).float()
    usess_itms = trans_to_cuda(usess_itms).long()

    alias_brnd_based_itms = trans_to_cuda(alias_brnd_based_itms).long()
    local_adj_brnd_based_itms = trans_to_cuda(local_adj_brnd_based_itms).float()
    ubrnd_based_itms = trans_to_cuda(ubrnd_based_itms).long()

    alias_cat_based_itms = trans_to_cuda(alias_cat_based_itms).long()
    local_adj_cat_based_itms = trans_to_cuda(local_adj_cat_based_itms).float()
    ucat_based_itms = trans_to_cuda(ucat_based_itms).long()

    alias_sess_brnds = trans_to_cuda(alias_sess_brnds).long()
    local_adj_brnds = trans_to_cuda(local_adj_brnds).float()
    usess_brnds = trans_to_cuda(usess_brnds).long()

    alias_sess_cats = trans_to_cuda(alias_sess_cats).long()
    local_adj_cats = trans_to_cuda(local_adj_cats).float()
    usess_cats = trans_to_cuda(usess_cats).long()

    alias_brnd_itms = trans_to_cuda( alias_brnd_itms).long()
    alias_itm_brnds = trans_to_cuda(alias_itm_brnds).long()
    local_adj_itm_brnd = trans_to_cuda(local_adj_itm_brnd).float()
    usess_itm_brnd = trans_to_cuda(usess_itm_brnd).long()

    alias_cat_itms = trans_to_cuda( alias_cat_itms).long()
    alias_itm_cats = trans_to_cuda(alias_itm_cats).long()
    local_adj_itm_cat = trans_to_cuda(local_adj_itm_cat).float()
    usess_itm_cat = trans_to_cuda(usess_itm_cat).long()

    local_itm_emb, local_brnd_based_itm_emb, local_cat_based_itm_emb, local_brnd_emb,\
        local_cat_emb, local_itm_brnd_emb, local_itm_cat_emb = model(sess_idxs, mask, sess_itms, 
                                        usess_itms, local_adj_itms, ubrnd_based_itms, local_adj_brnd_based_itms,
                                                ucat_based_itms, local_adj_cat_based_itms, usess_brnds, local_adj_brnds,
                                                               usess_cats, local_adj_cats, usess_itm_brnd, local_adj_itm_brnd,
                                                                                               usess_itm_cat, local_adj_itm_cat)

    get1 = lambda i: local_itm_emb[i][alias_sess_itms[i]]      
    locl_itm_emb = torch.stack([get1(i) for i in torch.arange(len(alias_sess_itms)).long()])

    get2 = lambda i: local_brnd_based_itm_emb[i][alias_brnd_based_itms[i]]      
    local_brnd_based_itm_emb = torch.stack([get2(i) for i in torch.arange(len(alias_brnd_based_itms)).long()])

    get2 = lambda i: local_cat_based_itm_emb[i][alias_cat_based_itms[i]]      
    local_cat_based_itm_emb = torch.stack([get2(i) for i in torch.arange(len(alias_cat_based_itms)).long()])
    
    get3 = lambda i: local_brnd_emb[i][alias_sess_brnds[i]] 
    locl_brnd_emb = torch.stack([get3(i) for i in torch.arange(len(alias_sess_brnds)).long()])

    get3 = lambda i: local_cat_emb[i][alias_sess_cats[i]] 
    locl_cat_emb = torch.stack([get3(i) for i in torch.arange(len(alias_sess_cats)).long()])
    
    get1_mix = lambda i: local_itm_brnd_emb[i][alias_brnd_itms[i]] 
    locl_brnd_itm_emb = torch.stack([get1_mix(i) for i in torch.arange(len(alias_brnd_itms)).long()])
    
    get2_mix = lambda i: local_itm_brnd_emb[i][alias_itm_brnds[i]]
    locl_itm_brnd_emb = torch.stack([get2_mix(i) for i in torch.arange(len(alias_itm_brnds)).long()])

    get3_mix = lambda i: local_itm_cat_emb[i][alias_cat_itms[i]] 
    locl_cat_itm_emb = torch.stack([get3_mix(i) for i in torch.arange(len(alias_cat_itms)).long()])
    
    get4_mix = lambda i: local_itm_cat_emb[i][alias_itm_cats[i]]
    locl_itm_cat_emb = torch.stack([get4_mix(i) for i in torch.arange(len(alias_itm_cats)).long()])
    
    scores = model.compute_scores(locl_itm_emb, local_brnd_based_itm_emb, local_cat_based_itm_emb,
                                                  locl_brnd_emb, locl_cat_emb, locl_brnd_itm_emb, locl_cat_itm_emb,
                                                                                   locl_itm_brnd_emb, locl_itm_cat_emb, mask)

    return targets, scores


def train_test(model, train_data, test_data):

    print('.... Training Step ....')
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=model.batch_size, shuffle=True, pin_memory=True)
    c=0
    for bch_data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, scores = forward(model, bch_data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()
    
    print('.... Evaluating Step ....')    
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=model.batch_size, shuffle=False, pin_memory=True)
    result = []
    hit_k10, mrr_k10, hit_k20, mrr_k20, hit_k30, mrr_k30, hit_k40, mrr_k40, hit_k50, mrr_k50 = [], [], [], [], [], [], [], [], [], []
    
    for data in test_loader:
        targets, scores = forward(model, data)
        sub_scores_k20 = scores.topk(20)[1]
        sub_scores_k20 = trans_to_cpu(sub_scores_k20).detach().numpy()
        sub_scores_k10 = scores.topk(10)[1]
        sub_scores_k10 = trans_to_cpu(sub_scores_k10).detach().numpy()
        
        sub_scores_k30 = scores.topk(30)[1]
        sub_scores_k30 = trans_to_cpu(sub_scores_k30).detach().numpy()
        sub_scores_k40 = scores.topk(40)[1]
        sub_scores_k40 = trans_to_cpu(sub_scores_k40).detach().numpy()
        sub_scores_k50 = scores.topk(50)[1]
        sub_scores_k50 = trans_to_cpu(sub_scores_k50).detach().numpy()
        targets = targets.numpy()
        
        for score, target, mask in zip(sub_scores_k20, targets, test_data.mask):
            hit_k20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k20.append(0)
            else:
                mrr_k20.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k10, targets, test_data.mask):
            hit_k10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k10.append(0)
            else:
                mrr_k10.append(1 / (np.where(score == target - 1)[0][0] + 1))
                
        for score, target, mask in zip(sub_scores_k30, targets, test_data.mask):
            hit_k30.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k30.append(0)
            else:
                mrr_k30.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k40, targets, test_data.mask):
            hit_k40.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k40.append(0)
            else:
                mrr_k40.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k50, targets, test_data.mask):
            hit_k50.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k50.append(0)
            else:
                mrr_k50.append(1 / (np.where(score == target - 1)[0][0] + 1))                
    
    result.append(np.mean(hit_k10) * 100)
    result.append(np.mean(mrr_k10) * 100)
    result.append(np.mean(hit_k20) * 100)
    result.append(np.mean(mrr_k20) * 100)
    
    result.append(np.mean(hit_k30) * 100)
    result.append(np.mean(mrr_k30) * 100)
    result.append(np.mean(hit_k40) * 100)
    result.append(np.mean(mrr_k40) * 100)
    result.append(np.mean(hit_k50) * 100)
    result.append(np.mean(mrr_k50) * 100)   
    return result