from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import re
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-dataset',type=str) #name of dataset which is used to generate candidate words
parser.add_argument('-path_pubmed',type=str) #path to csv file containing pubmed data
parser.add_argument('-path_data',type=str) #path to csv file containing dataset
parser.add_argument('-csv_path',type=str) #path to save the csv file
parser.add_argument('-reference_column',type=str) #column name in dataset containing reference summary

args = parser.parse_args()
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

train_data = pd.read_csv(args.path_data)
train_data = '\n'.join(train_data[args.reference_column].to_list()).lower()
train_data = re.sub(r'[^\w\s]', ' ', train_data)

train_pubmed = pd.read_csv(args.path_pubmed)
train_pubmed = '\n'.join(train_pubmed['text'].to_list()).lower()
train_pubmed = re.sub(r'[^\w\s]', ' ', train_pubmed)

def checkAllNum(chars):
    a= re.search('[0-9]',chars)
    b = ')' in chars or '(' in chars
    if a or b: return True
    return False

train_data = train_data.split()
train_data = [tok for tok in train_data if not checkAllNum(tok)]

train_pubmed = train_pubmed.split()
train_pubmed = [tok for tok in train_pubmed if not checkAllNum(tok)]

train_data

from collections import Counter
freq_toks = Counter(train_data)
print(len(freq_toks))

freq_pubmed = Counter(train_pubmed)
print(len(freq_pubmed))

import pandas as pd
import re

list_datatype, list_toks, list_splits,list_freq = list(), list(), list(), list()

for tok in freq_toks:
    tokenized = tokenizer_bert.tokenize(tok)
    if len(tokenized) == 1: continue
    list_datatype.append(args.dataset)
    list_toks.append(tok)
    list_splits.append(len(tokenized))
    list_freq.append(freq_toks[tok])
        

for tok in freq_pubmed:
    tokenized = tokenizer_bert.tokenize(tok)
    if len(tokenized) == 1: continue
    list_datatype.append('PubMed')
    list_toks.append(tok)
    list_splits.append(len(tokenized))
    list_freq.append(freq_pubmed[tok])

df = pd.DataFrame({'Token':list_toks, \
                   'SplitSize': list_splits, \
                   'Frequency': list_freq, \
                   'TokenFrom': list_datatype})

df.to_csv(args.csv_path,index=False)