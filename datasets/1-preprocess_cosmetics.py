import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timezone, timedelta


def process_seqs(tseqs, tdates):
    idx = 0
    out_seqs = []
    out_dates = []
    labs = []
    out_sidx = []
    for seq, date in zip(tseqs, tdates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            out_sidx += [idx]
            idx += 1
    return out_sidx, out_seqs, labs, out_dates


def process_test(test_sidx):
    idx = 0
    test_ids = []
    test_seqs = []
    test_dates = []
    for test_idx in test_sidx:
        temp_items, temp_time = session_data_map.get(test_idx)
        out_itm = []
        for itm, time in zip(temp_items, temp_time):         
            if itm in item_dict:             # Remove new item that dosent exist in trian set
                out_itm += [item_dict[itm]]   # Found the number that assigned earlier in item_dict
        if len(out_itm) < 2:    # May happen
            continue     
        test_ids += [idx]
        test_seqs += [out_itm]
        test_dates += [time]
        idx += 1
    return test_ids, test_seqs, test_dates


def process_train(train_sidx):
    idx = 0
    train_ids = []
    train_seqs = []
    train_dates = []
    itm_ctr = 0     # Count items
    for train_idx in train_sidx:
        temp_items, temp_time = session_data_map.get(train_idx)
        out_itm = []
        for itm, time in zip(temp_items, temp_time):
            cat = cat_dict.get(itm)
            brnd = brand_dict.get(itm)
            if itm in item_dict:
                out_itm += [item_dict[itm]]
            else:
                itm_ctr += 1
                item_dict[itm] = itm_ctr  # Add this new number to item dict.
                out_itm += [item_dict[itm]]         
                if itm_ctr not in new_item_category:
                    new_item_category[itm_ctr] = cat
                if itm_ctr not in new_item_brand:
                    new_item_brand[itm_ctr] = brnd

        train_ids += [idx]
        train_seqs += [out_itm]
        train_dates += [time]
        idx += 1
    print('\nItem_ctr:', itm_ctr)
    
    change = {}                      
    new_ctr =  itm_ctr   
    for k,v in new_item_category.items():
        if v in change:
            new_item_category[k] = change[v]
        else:
            new_ctr += 1
            change[v] = new_ctr
            new_item_category[k] = change[v]
    cat_ctr = new_ctr - (itm_ctr)
    print("\nCategory_ctr:", cat_ctr)   
    
    change = {}                      
    new_ctr = itm_ctr + cat_ctr  
    for k,v in new_item_brand.items():
        if v in change:
            new_item_brand[k] = change[v]
        else:
            new_ctr += 1
            change[v] = new_ctr
            new_item_brand[k] = change[v]
    brand_ctr = new_ctr - (itm_ctr+cat_ctr)
    print("\nBrand_ctr:", brand_ctr)
    
    return train_ids, train_seqs, train_dates, new_item_brand, new_item_category


test_days = 1
min_session_length = 2
max_session_length = 40
min_item_support = 5 

item_dict = {}
brand_dict = {}
cat_dict = {}
session_data_map = {}
new_item_brand = {}
new_item_category = {}

print('-- Loading data --')
data = pd.read_csv('raw/cosmetics/2019-Oct.csv', header=0, sep=',')
data = data[data['event_type'] == "view"].copy()
data['event_time'] = data['event_time'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S %Z").timestamp())
data['event_time'] = data['event_time'].astype('int')

print('-- Assigning session ids --') 
data.sort_values(by=['user_id', 'event_time'], ascending=True, inplace=True) # Sort data by user ID and time
new_user = data['user_id'].values[1:] != data['user_id'].values[:-1]
new_user = np.r_[True, new_user]
session_ids = np.cumsum(new_user) # Compute the session ids
data['session_id'] = session_ids

# =============================================================================
#           Filtering data
# =============================================================================
print("-- Filtering data --")
# Filter item support
item_supports = data.groupby('product_id').size()
data = data[np.in1d(data.product_id, item_supports[item_supports>= min_item_support].index)]

# Filter min session length
session_lengths = data.groupby('session_id').size()
data = data[np.in1d(data.session_id, session_lengths[session_lengths >= min_session_length].index)]

# Filter max length
session_lengths = data.groupby('session_id').size()
data = data[np.in1d(data.session_id, session_lengths[session_lengths <= max_session_length].index)]

# =============================================================================
#           Making sessions
# =============================================================================
print("-- Making sessions --")
SESSION_KEY = data.columns.get_loc('session_id')
ITEM_KEY = data.columns.get_loc('product_id')
TIME_KEY = data.columns.get_loc('event_time')
ITEM_CATEGORY = data.columns.get_loc('category_id')
ITEM_BRAND = data.columns.get_loc('brand') 

new_brand = {}
brnd_id = 1
temp_id = -1
temp_time = []
temp_item = []
for row in data.itertuples(index=False):
    itm = row[ITEM_KEY]
    time = row[TIME_KEY]
    if temp_id != row[SESSION_KEY]:
        session_data_map.update({temp_id : [temp_item, temp_time]})
        # Now clear temp lists
        temp_id = row[SESSION_KEY]
        temp_item = []
        temp_time = []
    temp_item.append(itm)
    temp_time.append(time)
    
    brnd = row[ITEM_BRAND]      # Such as 'nan', 'kapous', 'oniq', 'freedecor'       
    if itm not in brand_dict:
        brand_dict[itm] = brnd 
        
    cat = row[ITEM_CATEGORY]
    if itm not in cat_dict:
        cat_dict[itm] = cat 
session_data_map.update({temp_id : [temp_item, temp_time]})

# =============================================================================
#           Splitting train and test sessions
# =============================================================================
print("-- Splitting train set and test set --")
data_end = datetime.fromtimestamp(data['event_time'].max(), timezone.utc)
test_from = data_end - timedelta(test_days)
session_max_times = (data.groupby(['session_id'])['event_time']).max()   

train_keep = session_max_times[session_max_times < test_from.timestamp()].index
train = data[data['session_id'].isin(train_keep)]   
train_sidx = train['session_id']
train_sidx = np.unique(train_sidx)

test_keep = session_max_times[session_max_times >= test_from.timestamp()].index
test = data[data['session_id'].isin(test_keep)]
test_sidx = test['session_id']
test_sidx = np.unique(test_sidx)
print('\nOriginal Train sessions:', len(train_sidx), '\nOriginal Test sessions:', len(test_sidx))

# =============================================================================
#           Final processing train and test sessions
# =============================================================================
train_ids, train_seqs, train_dates, new_item_brand, new_item_category = process_train(train_sidx)
test_ids, test_seqs, test_dates = process_test(test_sidx)
all_train_seq = (train_ids, train_seqs, ['Nothing'], train_dates)

all = 0
for seq in train_seqs:
    all += len(seq)
for seq in test_seqs:
    all += len(seq)
print('\nAvg session length: ', all/(len(train_seqs) + len(test_seqs) * 1.0))

tr_ids, tr_seqs, tr_labs, tr_dates = process_seqs(train_seqs, train_dates)
te_ids, te_seqs, te_labs, te_dates = process_seqs(test_seqs, test_dates)
processed_train = (tr_ids, tr_seqs, tr_labs, tr_dates)
processed_test = (te_ids, te_seqs, te_labs, te_dates)
print('\nTrain sessions:', len(tr_seqs))
print('\nTest sessions:', len(te_seqs))

# =============================================================================
#           Save data
# =============================================================================
if not os.path.exists('cosmetics'):
    os.makedirs('cosmetics')
pickle.dump(processed_train, open('cosmetics/train.txt', 'wb'))
pickle.dump(processed_test, open('cosmetics/test.txt', 'wb'))
pickle.dump(all_train_seq, open('cosmetics/all_train_seq.txt', 'wb'))
pickle.dump(new_item_brand,open('cosmetics/brand.txt', 'wb'))
pickle.dump(new_item_category,open('cosmetics/category.txt', 'wb'))