import argparse
import csv
import pickle
import operator
import os

def process_seqs(tseqs, tdates):
    idx = 0
    out_seqs = []
    out_dates = []
    labs = []
    ids = []
    for seq, date in zip(tseqs, tdates):
        for i in range(1, len(seq)):
            tar = seq[-i]
            labs += [tar]
            out_seqs += [seq[:-i]]
            out_dates += [date]
            ids += [idx]
            idx += 1
    return ids, out_seqs, labs, out_dates

def process_test(tes_sess):
# Convert test sessions to sequences, ignoring items that do not appear in training set
    idx = 0
    test_ids = []
    test_seqs = []
    test_dates = []
    for s, date in tes_sess:
        seq = sess_clicks[s]
        outseq = []
        for itm in seq:
            if itm in item_dict:
                outseq += [item_dict[itm]]
        if len(outseq) < 2:
            continue
        test_ids += [idx]
        test_dates += [date]
        test_seqs += [outseq]
        idx += 1
    return test_ids, test_seqs, test_dates

def process_train(tra_sess):
    """Convert training sessions to sequences and renumber items to start from 1"""
    idx = 0
    train_ids = []
    train_seqs = []
    train_dates = []
    itm_ctr = 0
    for s, date in tra_sess:
        seq = sess_clicks[s]
        outseq = []
        for itm in seq:
            if itm in item_dict:
                outseq += [item_dict[itm]]
            else:
                itm_ctr += 1
                outseq += [itm_ctr]
                item_dict[itm] = itm_ctr
                new_item_brand[itm_ctr] = item_brand[itm] 
                new_item_category[itm_ctr] = item_category[itm] 
        if len(outseq) < 2:  # Doesn't occur
            continue
        train_ids += [idx]
        train_seqs += [outseq]
        train_dates += [date]
        idx += 1
    print('\nItem_ctr:', itm_ctr)     # 40727
    
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
    print("\nCategory_ctr:", cat_ctr)   #711
    
    
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
    print("\nBrand_ctr:", brand_ctr)   # 4160
    
    return train_ids, train_seqs, train_dates, new_item_brand, new_item_category


test_seconds = 100
min_session_len = 2
max_session_len = 40
min_item_support = 5

item_dict = {}
new_item_brand = {}
new_item_category = {}

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tmall', help='dataset name')
opt = parser.parse_args()
print(opt)

'''Read from "dataset15" and separate the 120000 first lines as "tmall_data".'''
with open('raw/tmall/tmall_data.csv', 'w') as tmall_data:
    with open('raw/tmall/dataset15.csv', 'r') as tmall_file:
        header = tmall_file.readline()
        tmall_data.write(header)
        for line in tmall_file:
            data = line[:-1].split('\t')
            if int(data[2]) > 120000:
                break
            tmall_data.write(line)

# =============================================================================
#           Making sessions
# =============================================================================
print("\n-- Starting --")
with open('raw/tmall/tmall_data.csv', "r") as f:
    reader = csv.DictReader(f, delimiter='\t')
    sess_clicks = {}
    sess_date = {}
    curid = -1
    curdate = None
    for data in reader:
        sessid = int(data['SessionId'])
        if curdate and not curid == sessid:
            date = curdate
            sess_date[curid] = date
        curid = sessid
        item = int(data['ItemId'])
        curdate = float(data['Time'])

        if sessid in sess_clicks:
            sess_clicks[sessid] += [item]
        else:
            sess_clicks[sessid] = [item]
    date = float(data['Time'])
    sess_date[curid] = date

print("\n-- Reading data --")

# =============================================================================
#           Creating dictionaries
# =============================================================================
with open('raw/tmall/Tmall_brand.csv',"r") as f:
    reader = csv.DictReader(f)
    item_brand = {}
    for data in reader:
        item_id = int(data['item_id'])
        item_brand[item_id] = int(data['brand_id'])

with open('raw/tmall/Tmall_category.csv',"r") as f:
    reader = csv.DictReader(f)
    item_category = {}
    for data in reader:
        item_id = int(data['item_id'])
        item_category[item_id] = int(data['category_id'])

# =============================================================================
#           Filtering data
# =============================================================================
# Filter out length 1 sessions
for s in list(sess_clicks):
    if len(sess_clicks[s]) == 1:
        del sess_clicks[s]
        del sess_date[s]

# Count number of times each item appears
iid_counts = {}
for s in sess_clicks:
    seq = sess_clicks[s]
    for iid in seq:
        if iid in iid_counts:
            iid_counts[iid] += 1
        else:
            iid_counts[iid] = 1

sorted_counts = sorted(iid_counts.items(), key=operator.itemgetter(1))

length = len(sess_clicks)
for s in list(sess_clicks):
    curseq = sess_clicks[s]
    filseq = list(filter(lambda i: iid_counts[i] >= min_item_support, curseq))
    if len(filseq) < min_session_len or len(filseq) > max_session_len:
        del sess_clicks[s]
        del sess_date[s]
    else:
        sess_clicks[s] = filseq

# =============================================================================
#           Splitting train and test sessions
# =============================================================================
print("\n-- Splitting train set and test set --")
# Split out test set based on dates
dates = list(sess_date.items())
maxdate = dates[0][1]

for _, date in dates:
    if maxdate < date:
        maxdate = date

# the last of 100 seconds for test
splitdate = maxdate - test_seconds
tra_sess = filter(lambda x: x[1] < splitdate, dates)
tes_sess = filter(lambda x: x[1] > splitdate, dates)

# Sort sessions by date
tra_sess = sorted(tra_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
tes_sess = sorted(tes_sess, key=operator.itemgetter(1))     # [(session_id, timestamp), (), ]
print('\nOriginal Train sessions:', len(tra_sess))    # 186670    # 7966257
print('\nOriginal Test sessions:', len(tes_sess))    # 15979     # 15324

# =============================================================================
#           Final processing train and test sessions
# =============================================================================
train_ids, train_seqs, train_dates, new_item_brand, new_item_category = process_train(tra_sess)
test_ids, test_seqs, test_dates = process_test(tes_sess)
all_train_seq = (train_seqs, ['Nothing'], train_dates)

all = 0
for seq in train_seqs:
    all += len(seq)
for seq in test_seqs:
    all += len(seq)
print('\nAvg session length: ', all/(len(train_seqs) + len(test_seqs) * 1.0))

tr_ids, tr_seqs, tr_labs, tr_dates = process_seqs(train_seqs, train_dates)
te_ids, te_seqs, te_labs, te_dates = process_seqs(test_seqs, test_dates)
processed_train = (tr_seqs, tr_labs, tr_dates)
processed_test = (te_seqs, te_labs, te_dates)
print('\nTrain sessions:', len(tr_seqs))
print('\nTest sessions:', len(te_seqs))

# =============================================================================
#           Save data
# =============================================================================
if not os.path.exists('tmall'):
    os.makedirs('tmall')
pickle.dump(processed_train, open('tmall/train.txt', 'wb'))
pickle.dump(processed_test, open('tmall/test.txt', 'wb'))
pickle.dump(all_train_seq, open('tmall/all_train_seq.txt', 'wb'))
pickle.dump(new_item_brand,open('tmall/brand.txt', 'wb'))
pickle.dump(new_item_category,open('tmall/category.txt', 'wb'))
