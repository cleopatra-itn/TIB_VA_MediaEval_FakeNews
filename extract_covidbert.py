import sys, os
import json, re
import pandas as pd
import numpy as np
import string

import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from helper_funcs import *
from preprocess_covidbert import preprocess_bert

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, root, csv_file, transform=None):
        self.file_names = pd.read_csv(os.path.join(root,csv_file))
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = str(self.file_names.iloc[idx,0])
        label = str(self.file_names.iloc[idx,1])

        text = open(os.path.join(self.root, 'data', fname+'.txt'), 'r', encoding='utf-8', errors='ignore').read().strip().lower()
    
        if self.transform:
            text = self.transform(text)    

        return fname, text, label


class SplitDataset(Dataset):
    def __init__(self, root, csv_file, data_dict):
        self.file_names = pd.read_csv(os.path.join(root,csv_file), header=None, delimiter=',')
        self.data_dict = data_dict
        self.root = root

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        ind = self.file_names.iloc[idx,0]

        text = self.data_dict[str(ind)]

        return text


dev_loc = 'dataset/dev/data/'
test_loc = 'dataset/test/data/'

## Dev Data Feature Extraction
data_dict = json.load(open(os.path.join(dev_loc, 'dev_data.json'), 'r', encoding='utf-8'))
for bert_type in [(AutoModel,    AutoTokenizer,    'digitalepidemiologylab/covid-twitter-bert-v2')]:
    tokenizer = bert_type[1].from_pretrained(bert_type[2], do_lower_case=True)
    model = bert_type[0].from_pretrained(bert_type[2], output_hidden_states=True)
    model.to(device).eval()
    
    embed_dict = {'sent_word_catavg':[], 'sent_word_sumavg': [], 'sent_emb_2_last': [],
                        'sent_emb_last': []}


    ## Pass preprocess_bert to SplitDataset for COVID Twitter-BERT preprocessing
    ph_data = SplitDataset(dev_loc, 'all_ids.csv', data_dict)
    ## Batch size 1 because not enforcing same size (# tokens) to each tweet
    ph_loader = DataLoader(ph_data, batch_size=1, sampler=SequentialSampler(ph_data))

    for i, batch in enumerate(ph_loader):
        print(i)
        text = batch

        sent_word_catavg, sent_word_sumavg, sent_emb_2_last, sent_emb_last, \
            cls_out = get_word_sent_embedding(text[0], model, tokenizer, device)
        
        embed_dict['sent_word_catavg'].append(sent_word_catavg.tolist())
        embed_dict['sent_word_sumavg'].append(sent_word_sumavg.tolist())
        embed_dict['sent_emb_2_last'].append(sent_emb_2_last.tolist())
        embed_dict['sent_emb_last'].append(sent_emb_last.tolist())

    json.dump(embed_dict, open('features/dev_covidbert.json', 'w'))



## Test Data Feature Extraction
data_dict = json.load(open(os.path.join(test_loc, 'test_data.json'), 'r', encoding='utf-8'))
for bert_type in [(AutoModel,    AutoTokenizer,    'digitalepidemiologylab/covid-twitter-bert-v2')]:
    tokenizer = bert_type[1].from_pretrained(bert_type[2], do_lower_case=True)
    model = bert_type[0].from_pretrained(bert_type[2], output_hidden_states=True)
    model.to(device).eval()
    
    embed_dict = {'sent_word_catavg':[], 'sent_word_sumavg': [], 'sent_emb_2_last': [],
                        'sent_emb_last': []}


    test_ids = pd.read_csv('dataset/test/tweets/test_tweets.txt', header=None)[0].to_numpy().flatten().astype(str)

    for i, tid in enumerate(test_ids):
        print(i)
        if tid in data_dict:
            text = data_dict[tid]
#             text = preprocess_bert(text) ## Uncomment this to process text using COVID Twitter-BERT preprocessing

            sent_word_catavg, sent_word_sumavg, sent_emb_2_last, sent_emb_last, \
                cls_out = get_word_sent_embedding(text, model, tokenizer, device)
        else:
            sent_word_sumavg, sent_word_catavg, sent_emb_2_last, sent_emb_last = [], [], [], []
            
        embed_dict['sent_word_catavg'].append(sent_word_catavg)
        embed_dict['sent_word_sumavg'].append(sent_word_sumavg)
        embed_dict['sent_emb_2_last'].append(sent_emb_2_last)
        embed_dict['sent_emb_last'].append(sent_emb_last)

    json.dump(embed_dict, open('features/test_covidbert.json', 'w'))
