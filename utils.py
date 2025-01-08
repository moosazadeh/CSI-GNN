import numpy as np
import torch
from torch.utils.data import Dataset

def process_data(all_sess_itms):
    all_sess_lens = [len(sess_itms) for sess_itms in all_sess_itms]
    max_len = max(all_sess_lens)
        
    # To compute the position attention, reverse each session.
    all_sess_itms = [list(reversed(sess_itms)) + [0] * (max_len - sess_len) if sess_len < max_len else list(reversed(sess_itms[-max_len:]))
               for sess_itms, sess_len in zip(all_sess_itms, all_sess_lens)]

    # Create mask of each session
    mask = [[1] * sess_len + [0] * (max_len - sess_len) if sess_len < max_len else [1] * max_len
               for sess_len in all_sess_lens]
    return all_sess_itms, mask, max_len

def process_brnds(brand, sess_itms):  
    sess_brnds = []
    for item in sess_itms:
       if item == 0:
          sess_brnds += [0]
       else:
          sess_brnds += [brand[item]]
    return sess_brnds

def process_cats(category, sess_itms):  
    sess_cats = []
    for item in sess_itms:
       if item == 0:
          sess_cats += [0]
       else:
          sess_cats += [category[item]]
    return sess_cats

class Data(Dataset):
    def __init__(self, data, brand, category):
        all_sess_itms, mask, max_len = process_data(data[1])
        self.brand = brand
        self.category = category
        self.all_sess_itms = np.asarray(all_sess_itms)
        self.targets = np.asarray(data[2])
        self.mask = np.asarray(mask)
        self.length = len(data[1])
        self.max_len = max_len

    def __len__(self):
        return self.length
    
    def __getitem__(self, sess_idxs):
        max_n_itm = self.max_len
        sess_itms, mask, target = self.all_sess_itms[sess_idxs], self.mask[sess_idxs], self.targets[sess_idxs]
        sess_brnds = process_brnds(self.brand, sess_itms)
        sess_cats = process_cats(self.category, sess_itms)
        
        sess_itm_brnd = np.append(sess_itms, sess_brnds)
        sess_itm_brnd = sess_itm_brnd[sess_itm_brnd > 0]
        
        sess_itm_cat = np.append(sess_itms, sess_cats)
        sess_itm_cat = sess_itm_cat[sess_itm_cat > 0]
        
        uitms = np.unique(sess_itms)
        ubrnds = np.unique(sess_brnds)
        ucats = np.unique(sess_cats)
        
        uitm_brnd = np.unique(sess_itm_brnd)
        if len(uitm_brnd)<max_n_itm*2:
          uitm_brnd= np.append(uitm_brnd,0)
        
        uitm_cat = np.unique(sess_itm_cat)
        if len(uitm_cat)<max_n_itm*2:
          uitm_cat= np.append(uitm_cat,0)
          
        usess_itms = uitms.tolist() + (max_n_itm - len(uitms)) * [0]
        usess_brnds = ubrnds.tolist()  + (max_n_itm - len(ubrnds)) * [0]
        usess_cats = ucats.tolist()  + (max_n_itm - len(ucats)) * [0]
        usess_itm_brnd = uitm_brnd.tolist() + (max_n_itm*2 - len(uitm_brnd)) * [0]
        usess_itm_cat = uitm_cat.tolist() + (max_n_itm*2 - len(uitm_cat)) * [0]

        adj_itms = np.zeros((max_n_itm, max_n_itm))
        adj_brnds = np.zeros((max_n_itm, max_n_itm))
        adj_cats = np.zeros((max_n_itm, max_n_itm))
        adj_itm_brnd = np.zeros((max_n_itm*2, max_n_itm*2))
        adj_itm_cat = np.zeros((max_n_itm*2, max_n_itm*2))

        '''Define different types of edges between two nodes into matrix "adj_itms":
                1=self_loop,  2=in(<--),  3=out(-->),  4=in-out(<-->) '''

# =====================================================================================================================
#       Define Item-Item relations as a graph.
# =====================================================================================================================
        for i in np.arange(len(sess_itms) - 1):
            v1 = np.where(uitms == sess_itms[i])[0][0]
            adj_itms[v1][v1] = 1
            if sess_itms[i + 1] == 0:
                break
            v2 = np.where(uitms == sess_itms[i + 1])[0][0]
            if v1 == v2 or adj_itms[v1][v2] == 4:
                continue
            adj_itms[v2][v2] = 1
            if adj_itms[v2][v1] == 2:
                adj_itms[v1][v2] = 4
                adj_itms[v2][v1] = 4
            else:
                adj_itms[v1][v2] = 2
                adj_itms[v2][v1] = 3
        # alias_sess_itms represents the relative position of the items clicked sequentially in each session.
        alias_sess_itms = [np.where(uitms == itm)[0][0] for itm in sess_itms]

# =====================================================================================================================
#       Define brand-based-Item-Item relations as a graph.
# =====================================================================================================================
        adj_brnd_based_itms = np.zeros((max_n_itm, max_n_itm))
        sub_seqs ={}
        brnd_based_itms = []       # sess_itms in new order
        alias_brnd_based_itms = []
        for i in np.arange(len(sess_itms) - 1):
            brnd = sess_brnds[i]
            if brnd not in sub_seqs:
                sub_seqs[brnd] = [sess_itms[i]]
            else:
                sub_seqs[brnd] += [sess_itms[i]]
        
        for brnd, seq_itms in sub_seqs.items():
            for i in np.arange(len(seq_itms) - 1):
                v1 = np.where(uitms == seq_itms[i])[0][0]    
                adj_brnd_based_itms[v1][v1] = 1
                if seq_itms[i + 1] == 0:
                    break
                v2 = np.where(uitms == seq_itms[i + 1])[0][0]
                if v1 == v2 or adj_brnd_based_itms[v1][v2] == 4:
                    continue
                adj_brnd_based_itms[v2][v2] = 1
                if adj_brnd_based_itms[v2][v1] == 2:
                    adj_brnd_based_itms[v1][v2] = 4
                    adj_brnd_based_itms[v2][v1] = 4
                else:
                    adj_brnd_based_itms[v1][v2] = 2
                    adj_brnd_based_itms[v2][v1] = 3
            brnd_based_itms += seq_itms
        
        usub_itms = np.unique(brnd_based_itms)
        ubrnd_based_itms = usub_itms.tolist() + (max_n_itm - len(usub_itms)) * [0]
        
        # alias_sess_itms represents the relative position of the items clicked sequentially in each session.
        for itm in brnd_based_itms:
            alias_brnd_based_itms += [np.where(usub_itms == itm)[0][0]]
        if len(alias_brnd_based_itms) < self.max_len:
            alias_brnd_based_itms += [0]

# =====================================================================================================================
#       Define category-based-Item-Item relations as a graph.
# =====================================================================================================================
        adj_cat_based_itms = np.zeros((max_n_itm, max_n_itm))
        sub_seqs ={}
        cat_based_itms = []       # sess_itms in new order
        alias_cat_based_itms = []
        for i in np.arange(len(sess_itms) - 1):
            cat = sess_cats[i]
            if cat not in sub_seqs:
                sub_seqs[cat] = [sess_itms[i]]
            else:
                sub_seqs[cat] += [sess_itms[i]]
        
        for cat, seq_itms in sub_seqs.items():
            for i in np.arange(len(seq_itms) - 1):
                v1 = np.where(uitms == seq_itms[i])[0][0]    
                adj_cat_based_itms[v1][v1] = 1
                if seq_itms[i + 1] == 0:
                    break
                v2 = np.where(uitms == seq_itms[i + 1])[0][0]
                if v1 == v2 or adj_cat_based_itms[v1][v2] == 4:
                    continue
                adj_cat_based_itms[v2][v2] = 1
                if adj_cat_based_itms[v2][v1] == 2:
                    adj_cat_based_itms[v1][v2] = 4
                    adj_cat_based_itms[v2][v1] = 4
                else:
                    adj_cat_based_itms[v1][v2] = 2
                    adj_cat_based_itms[v2][v1] = 3
            cat_based_itms += seq_itms
        
        usub_itms = np.unique(cat_based_itms)
        ucat_based_itms = usub_itms.tolist() + (max_n_itm - len(usub_itms)) * [0]
        
        # alias_sess_itms represents the relative position of the items clicked sequentially in each session.
        for itm in cat_based_itms:
            alias_cat_based_itms += [np.where(usub_itms == itm)[0][0]]
        if len(alias_cat_based_itms) < self.max_len:
            alias_cat_based_itms += [0]
            
# =====================================================================================================================
#       Define Item-Brand and Brand-Item relations as two graphs.
# =====================================================================================================================
        for i in np.arange(len(sess_itms) - 1):
            v1 = np.where(uitm_brnd == sess_itms[i])[0][0]
            b1 = np.where(uitm_brnd == self.brand[sess_itms[i]])[0][0]
            adj_itm_brnd[v1][v1] = 1
            adj_itm_brnd[b1][b1] = 4
            adj_itm_brnd[v1][b1] = 2
            adj_itm_brnd[b1][v1] = 3
            
            if sess_itms[i + 1] == 0:
                break
            v2 = np.where(uitm_brnd == sess_itms[i + 1])[0][0]
            b2 = np.where(uitm_brnd == self.brand[sess_itms[i + 1]])[0][0]
            adj_itm_brnd[v1][v2] = 1
            adj_itm_brnd[v2][v1] = 1
            adj_itm_brnd[b1][b2] = 4
            adj_itm_brnd[b2][b1] = 4
            
        alias_brnd_itm = [np.where(uitm_brnd == itm)[0][0] for itm in sess_itms]
        alias_itm_brnd = [np.where(uitm_brnd == brnd)[0][0] for brnd in sess_brnds]

# =====================================================================================================================
#       Define Item-Category and Category-Item relations as two graphs.
# =====================================================================================================================
        for i in np.arange(len(sess_itms) - 1):
            v1 = np.where(uitm_cat == sess_itms[i])[0][0]
            c1 = np.where(uitm_cat == self.category[sess_itms[i]])[0][0]
            adj_itm_cat[v1][v1] = 1
            adj_itm_cat[c1][c1] = 4
            adj_itm_cat[v1][c1] = 2
            adj_itm_cat[c1][v1] = 3
            
            if sess_itms[i + 1] == 0:
                break          
            v2 = np.where(uitm_cat == sess_itms[i + 1])[0][0]
            c2 = np.where(uitm_cat == self.category[sess_itms[i + 1]])[0][0]
            adj_itm_cat[v1][v2] = 1
            adj_itm_cat[v2][v1] = 1        
            adj_itm_cat[c1][c2] = 4
            adj_itm_cat[c2][c1] = 4
        
        alias_cat_itm = [np.where(uitm_cat == itm)[0][0] for itm in sess_itms]
        alias_itm_cat = [np.where(uitm_cat == cat)[0][0] for cat in sess_cats]
        
# =====================================================================================================================
#       Define Brand-Brand relations as a graph.
# =====================================================================================================================
        for i in np.arange(len(sess_brnds) - 1):
            if sess_brnds[i + 1] == 0:
                break
            b1 = np.where(ubrnds == sess_brnds[i])[0][0]
            b2 = np.where(ubrnds == sess_brnds[i + 1])[0][0]
            adj_brnds[b1][b2] += 1
        b_sum_in = np.sum(adj_brnds, 0)
        b_sum_in[np.where(b_sum_in == 0)] = 1
        adj_brnds_in = np.divide(adj_brnds, b_sum_in)
        b_sum_out = np.sum(adj_brnds, 1)
        b_sum_out[np.where(b_sum_out == 0)] = 1
        adj_brnds_out = np.divide(adj_brnds.transpose(), b_sum_out)
        adj_brnds = np.concatenate([adj_brnds_in, adj_brnds_out]).transpose()
        # alias_sess_brnds represents the relative position of the brands clicked sequentially in each session.
        alias_sess_brnds = [np.where(ubrnds == brnd)[0][0] for brnd in sess_brnds]

# =====================================================================================================================
#       Define Category-Category relations as a graph.
# =====================================================================================================================
        for i in np.arange(len(sess_cats) - 1):
            if sess_cats[i + 1] == 0:
                break
            c1 = np.where(ucats == sess_cats[i])[0][0]
            c2 = np.where(ucats == sess_cats[i + 1])[0][0]
            adj_cats[c1][c2] += 1
        c_sum_in = np.sum(adj_cats, 0)
        c_sum_in[np.where(c_sum_in == 0)] = 1
        adj_cats_in = np.divide(adj_cats, c_sum_in)
        c_sum_out = np.sum(adj_cats, 1)
        c_sum_out[np.where(c_sum_out == 0)] = 1
        adj_cats_out = np.divide(adj_cats.transpose(), c_sum_out)
        adj_cats = np.concatenate([adj_cats_in, adj_cats_out]).transpose()
        # alias_sess_cats represents the relative position of the categories clicked sequentially in each session in the "category" list of this session.
        alias_sess_cats = [np.where(ucats == cat)[0][0] for cat in sess_cats]




      
        return [torch.tensor(sess_idxs), torch.tensor(mask), torch.tensor(target), torch.tensor(sess_itms),
                torch.tensor(alias_sess_itms), torch.tensor(adj_itms), torch.tensor(usess_itms),
                torch.tensor(alias_brnd_based_itms), torch.tensor(adj_brnd_based_itms), torch.tensor(ubrnd_based_itms),
                torch.tensor(alias_cat_based_itms), torch.tensor(adj_cat_based_itms), torch.tensor(ucat_based_itms),
                torch.tensor(alias_sess_brnds), torch.tensor(adj_brnds), torch.tensor(usess_brnds),
                torch.tensor(alias_sess_cats), torch.tensor(adj_cats), torch.tensor(usess_cats),
                torch.tensor(alias_brnd_itm), torch.tensor(alias_itm_brnd), torch.tensor(adj_itm_brnd), torch.tensor(usess_itm_brnd),
                torch.tensor(alias_cat_itm), torch.tensor(alias_itm_cat), torch.tensor(adj_itm_cat), torch.tensor(usess_itm_cat)]
    
    
    