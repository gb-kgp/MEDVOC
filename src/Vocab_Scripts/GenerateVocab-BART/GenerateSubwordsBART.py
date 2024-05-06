#!/usr/bin/env python
# coding: utf-8
import argparse
from transformers import BartTokenizer
from tokenizers import ByteLevelBPETokenizer
import os
import pickle as pkl
import json
import pandas as pd

if not os.path.exists('VocabFiles_BART'): os.mkdir('VocabFiles_BART')

parser = argparse.ArgumentParser()
parser.add_argument('-v_size', type=int,required=False)
parser.add_argument('-dataset',type=str,required=True)
parser.add_argument('-frac',type=float,required=False)
parser.add_argument('-csv_path',type=str,required=True)


args = parser.parse_args()

df_PM = pd.read_csv(args.csv_path)
df_consider = df_PM[df_PM['Consider']==1]
df_PM = df_consider[df_consider['TokenFrom']=='PubMed']

df_TGT = pd.read_csv(args.csv_path)
df_consider = df_TGT[df_TGT['Consider']==1]
df_TGT = df_consider[df_consider['TokenFrom']==args.dataset]

list_PM_All = list()
for idx in range(df_PM.shape[0]):
    try: list_PM_All.append(' '.join([df_PM.iloc[idx,0]]*df_PM.iloc[idx,2]))
    except: print('Error:',df_PM.iloc[idx,0],df_PM.iloc[idx,2])

with open('VocabFiles_BART/PM_All','w') as f:
     f.write('\n'.join(list_PM_All))
f.close()

list_TGT_All = list()
for idx in range(df_TGT.shape[0]):
    list_TGT_All.append(' '.join([df_TGT.iloc[idx,0]]*df_TGT.iloc[idx,2]))

with open(f'VocabFiles_BART/{args.dataset}_All','w') as f:
     f.write('\n'.join(list_TGT_All))
f.close()

pretrained_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
pretrained_vocab = pretrained_tokenizer.get_vocab()

domain_tokenizer_PM_All = ByteLevelBPETokenizer()
domain_tokenizer_PM_All.train('VocabFiles_BART/PM_ALL',vocab_size=50000)

domain_tokenizer_TGT_All = ByteLevelBPETokenizer()
domain_tokenizer_TGT_All.train(f'VocabFiles_BART/{args.dataset}_All')

def get_domain_traits(domain_tokenizer, vocab_path):
    
    if not os.path.exists(vocab_path): os.mkdir(vocab_path)
    
    domain_tokenizer.save_model(vocab_path, prefix="custom")
    custom_merges_path = os.path.join(vocab_path,"custom-merges.txt")

    domain_vocab = domain_tokenizer.get_vocab()
    sorted_bpe = sorted(domain_vocab.items(), key=lambda x: x[-1])
    
    with open(custom_merges_path, "r") as reader:
        merges_file = reader.readlines()
    start_index = len(sorted_bpe) - len(merges_file) + 1
    vocab_merge_pair = []

    for bpe, merge in zip(sorted_bpe[start_index:], merges_file[1:]):
        vocab_merge_pair.append({"vocab": bpe, "merge": merge})

    vocab_merges = vocab_merge_pair

    complement_pairs = [example for example in vocab_merges if
                        example["vocab"][0] not in pretrained_vocab]
    return complement_pairs

complement_pairs_PM_All = get_domain_traits(domain_tokenizer_PM_All,f'BART-Config/BART-PM-All-{args.v_size}-{args.dataset}')
complement_pairs_TGT_All = get_domain_traits(domain_tokenizer_TGT_All,f'BART-Config/BART-TGT-All-{args.v_size}-{args.dataset}')

def retain_common_set(cp1,cp2): #cp1: pm, cp2: tgt
    l_cp2 = [x for x in cp2]
    vocab_cp1 = [d['vocab'][0] for d in cp1[:args.v_size]]
    vocab_cp2 = [[idx,d['vocab'][0]] for idx,d in enumerate(cp2)]
    
    list_del = list()
    
    for idx,key in enumerate(vocab_cp2):
        if key[1] not in vocab_cp1: list_del.append(idx)
    
    for i in  list_del[::-1]: del l_cp2[i]
        
    return l_cp2

common_part_TGT_All = retain_common_set(complement_pairs_PM_All, complement_pairs_TGT_All)

print(len(complement_pairs_TGT_All), len(common_part_TGT_All))

vocab_common_TGT_All = [x['vocab'][0] for x in common_part_TGT_All]
check_terms_TGT_All = [x['merge'].strip().split() for x in common_part_TGT_All]
check_term_f_TGT_All = [x for sub in check_terms_TGT_All for x in sub if \
                   (x not in pretrained_vocab and x not in vocab_common_TGT_All)]

pairs_TGT_All = [x for x in common_part_TGT_All]

def add_term_TGT_All(term):
    if (term in vocab_common_TGT_All) or (term in pretrained_vocab) : 
        # if (term in vocab_common_TGT_All) : print(f'{term} in GT4: {vocab_common_TGT_All.index(term)}')
        # else: print(f'{term} in PRETRAINED_VOCAB: {pretrained_vocab[term]}')
        return
        
    for tup in complement_pairs_TGT_All:
        if tup['vocab'][0] == term:
            # print(tup)
            pairs_TGT_All.append(tup)
            vocab_common_TGT_All.append(term)
            
            m1,m2 = tup['merge'].strip().split()
            add_term_TGT_All(m1)
            add_term_TGT_All(m2)

for term in check_term_f_TGT_All:
    # print('----------')
    add_term_TGT_All(term)
    # print('----------')


print(len(pairs_TGT_All), len(vocab_common_TGT_All))

def return_PAC_A(PM, TGT,frac):
    PM_C = [x for x in PM]
    vocab_cp1 = [d['vocab'][0] for d in TGT]
    vocab_cp2 = [[idx,d['vocab'][0]] for idx,d in enumerate(PM)]
    
    list_del = list()
    
    for idx,key in enumerate(vocab_cp2):
        if key[1] in vocab_cp1: list_del.append(idx)
    
    for i in  list_del[::-1]: del PM_C[i]
        
    return PM_C[:int(frac*len(TGT))]

PM_all = return_PAC_A(complement_pairs_PM_All, pairs_TGT_All,args.frac)

pairs_PAC_All = [x for x in PM_all]
vocab_common_PAC_All = [x['vocab'][0] for x in PM_all]
check_terms_PAC_All = [x['merge'].strip().split() for x in PM_all]
check_term_PAC_f_All = [x for sub in check_terms_PAC_All for x in sub if \
                   (x not in pretrained_vocab and x not in vocab_common_PAC_All)]

# print(pairs_PAC_All, vocab_common_PAC_All, check_terms_PAC_All, check_term_PAC_f_All)

def add_term_PAC_All(term):
    if (term in vocab_common_PAC_All) or (term in pretrained_vocab): 
        # if (term in vocab_common_PAC_All) : print(f'{term} in ALL: {vocab_common_PAC_All.index(term)}')
        # else: print(f'{term} in PRETRAINED_VOCAB: {pretrained_vocab[term]}')
        return
    
    for tup in complement_pairs_PM_All:
        if tup['vocab'][0] == term:
            # print(tup)
            pairs_PAC_All.append(tup)
            vocab_common_PAC_All.append(term)
            
            m1,m2 = tup['merge'].strip().split()
            add_term_PAC_All(m1)
            add_term_PAC_All(m2)
        
print('**********************PAC-ALL************************')
for term in check_term_PAC_f_All:
    # print('--------')
    add_term_PAC_All(term)
    # print('-------------')

def return_union(cp1, cp2):
    cp1_c = [x for x in cp1]
    cp2_c = [x for x in cp2]
    vocab_cp1 = [d['vocab'][0] for d in cp1]
    vocab_cp2 = [[idx,d['vocab'][0]] for idx,d in enumerate(cp2)]
    
    list_del = list()
    
    for idx,key in enumerate(vocab_cp2):
        if key[1] in vocab_cp1: list_del.append(idx)
    
    for i in  list_del[::-1]: del cp2_c[i]
    
    cp1_c.extend(cp2_c)
    
    return cp1_c

complement_pairs =return_union(pairs_PAC_All,pairs_TGT_All)
# print(complement_pairs)


tok_save_path = f'BART-{args.dataset}/{args.v_size//1000}K-0'
pretrained_tokenizer.save_pretrained(tok_save_path)
original_vocab = pretrained_tokenizer.get_vocab()

import pickle as pkl
with open(f'{tok_save_path}/v_tgt.pkl', "wb") as writer:
    pkl.dump(pairs_TGT_All, writer)
writer.close()

with open(f'{tok_save_path}/v_ApPAC.pkl', "wb") as writer:
    pkl.dump(pairs_PAC_All, writer)
writer.close()

complement_pairs = sorted(pairs_TGT_All, key = lambda x:x['vocab'][1])

print('Final Vocab Size:',len(complement_pairs))

writer = open(f'{tok_save_path}/merges.txt', "a")
for vocab_merge_pair in complement_pairs:
    #print(vocab_merge_pair)
    vocab = vocab_merge_pair["vocab"]
    merge = vocab_merge_pair["merge"]
    original_vocab.update({vocab[0]: len(original_vocab)})
    writer.write(merge)
writer.close()

with open(f'{tok_save_path}/vocab.json', "w") as writer:
    json.dump(original_vocab, writer)


import pandas as pd
df = pd.read_csv(args.csv_path)
terms_EBM = df[df['TokenFrom']==args.dataset]['Token'].to_list()
freq_ebm = df[df['TokenFrom']==args.dataset]['Frequency'].to_list()
split_bart = df[df['TokenFrom'] == args.dataset]['SplitSize'].to_list()

sum_num = 0.
sum_den = 0.
for idx,term in enumerate(terms_EBM):
    sum_num += split_bart[idx]*freq_ebm[idx]
    sum_den += freq_ebm[idx]

print(sum_num/sum_den)

from transformers import BartTokenizer
import glob
from collections import defaultdict

dict_scores = defaultdict(lambda : defaultdict(dict))
for fname in sorted(glob.glob(f'./BART-{args.dataset}/*'),key = lambda x: [int(x.split('/')[-1].split('-')[-2][:-1]),float(x.split('/')[-1].split('-')[-1])]):
    domain_tok = BartTokenizer.from_pretrained(fname)
    sum_num = 0.
    sum_den = 0.
    
    for idx,term in enumerate(terms_EBM):
        sum_num += min(len(domain_tok.tokenize(term)),len(domain_tok.tokenize(' '+term)))*freq_ebm[idx]
        sum_den += freq_ebm[idx]

    print(fname, sum_num/sum_den, len(domain_tok))
    key = fname.split('/')[-1].split('-')
    dict_scores[key[-2]][key[-1]] = [round(sum_num/sum_den,2),len(domain_tok)]

for k1 in dict_scores:
    if k1 == '0K': continue
    print('data',end='\t')
    for k2 in dict_scores[k1]:
        print(k2,end='\t')
    print()
    break

for k1 in dict_scores:
    print(k1,end='\t')
    for k2 in dict_scores[k1]:
        print(f'{dict_scores[k1][k2][0]}',end='\t')
    print()
