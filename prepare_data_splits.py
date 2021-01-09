import numpy as np
from sklearn.model_selection import StratifiedKFold
import json
import os
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

dev_loc = 'dataset/dev/data/'
test_loc = 'dataset/test/data/'

## Development data
lab = 0
all_ids = []
all_labs = []
dev_data = {}
ext_data = {}
sent_lens = {}
for js in ['5g_corona_conspiracy.json', 'other_conspiracy.json', 'non_conspiracy.json']:
    js_file = json.load(open(dev_loc+js, 'r', encoding='utf-8'))

    for obj in js_file:
        temp = []
        idx = obj['id']
        text = obj['full_text']

        all_ids.append(idx)
        all_labs.append(lab)

        dev_data[idx] = text

        user = obj['user']
        temp.append(user['followers_count'])
        temp.append(user['friends_count'])
        temp.append(user['listed_count'])
        temp.append(user['favourites_count'])
        temp.append(0) if user['verified'] == 'false' else temp.append(1)
        temp.append(user['statuses_count'])
        temp.append(obj['retweet_count'])
        temp.append(obj['favorite_count'])

        if js not in sent_lens:
            sent_lens[js] = [len(text.split())]
        else:
            sent_lens[js].append(len(text.split()))

        ext_data[idx] = temp
    
    lab += 1


json.dump(ext_data, open(dev_loc+'dev_ext_data.json','w', encoding='utf-8'))

data_df = {'ids': all_ids, 'labs': all_labs}
data_df = pd.DataFrame(data_df)
data_df.to_csv(dev_loc+'all_ids.csv', header=None, index=None)
json.dump(dev_data, open(dev_loc+'dev_data.json','w', encoding='utf-8'))

all_labs = np.array(all_labs)

## Stratified 5-Fold Split
skf = StratifiedKFold(n_splits=5)

cnt = 1
for train_index, test_index in skf.split(range(len(all_ids)), all_labs):
    print(len(train_index), len(test_index))
    
    tr_ids = pd.DataFrame(train_index)
    te_ids = pd.DataFrame(test_index)
    tr_ids.to_csv(dev_loc+'splits/train%d.txt'%(cnt), header=None, index=None)
    te_ids.to_csv(dev_loc+'splits/val%d.txt'%(cnt), header=None, index=None)

    cnt += 1

