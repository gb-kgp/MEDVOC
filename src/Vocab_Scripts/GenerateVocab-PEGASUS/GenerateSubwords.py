#!/usr/bin/env python
import argparse
from transformers import PegasusTokenizer
import sentencepiece as spm
import pandas as pd
import os

if not os.path.exists('VocabFiles_PEGASUS'): os.mkdir('VocabFiles_PEGASUS')

parser = argparse.ArgumentParser()
parser.add_argument('-csv_path',type=str,required=True)
parser.add_argument('-v_size_TGT',type=int,required=True)
parser.add_argument('-v_size_PM',type=int,required=True)
parser.add_argument('-dataset',type=str,required=True)
parser.add_argument('-frac',type=float,required=True)

args = parser.parse_args()
csv_path = args.csv_path

df=pd.read_csv(csv_path)
df = df[df['Consider']==1]

##--PAC-SUmm Files
df_PM = df[df['TokenFrom']=='PubMed'].dropna()

df_PM_ALL = df_PM
df_PM_ALL = df_PM_ALL.drop(columns=['SplitSize','TokenFrom','Consider'])
df_PM_ALL.to_csv(f'VocabFiles_PEGASUS/{args.dataset}_PM_ALL_Freq.tsv',index=False,sep='\t',header=False)

##--TGT Files
df_TGT = df[df['TokenFrom']=='CHQ'].dropna()
df_TGT = df_TGT[df_TGT['Consider']==1]
df_TGT = df_TGT.dropna()

df_TGT_All = df_TGT.drop(columns=['SplitSize','TokenFrom','Consider'])
df_TGT_All.to_csv(f'VocabFiles_PEGASUS/{args.dataset}_All_Freq.tsv',index=False,sep='\t',header=False)


## -- Training Target
spm.SentencePieceTrainer.train(f'--input=VocabFiles_PEGASUS/{args.dataset}_All_Freq.tsv --input_format=tsv \
                               --model_prefix=VocabFiles_PEGASUS/{args.dataset}_All --vocab_size={args.v_size_TGT}')

spm.SentencePieceTrainer.train(f'--input=VocabFiles_PEGASUS/{args.dataset}_PM_ALL_Freq.tsv --input_format=tsv \
                               --model_prefix=VocabFiles_PEGASUS/{args.dataset}_PM_ALL --vocab_size=70000')


tok = PegasusTokenizer.from_pretrained('google/pegasus-large')
org_vocab = tok.get_vocab()

list_PM_All = list()

with open(f'VocabFiles_PEGASUS/{args.dataset}_PM_ALL.vocab','r') as f:
    for idx,line in enumerate(f):
        if idx < 4: continue
        term = line.split()[0]
        if term in org_vocab: continue
        list_PM_All.append(line)

list_TGT_All = list()
PAC_topK = [x.split()[0] for x in list_PM_All[:args.v_size_PM]]
with open(f'VocabFiles_PEGASUS/{args.dataset}_All.vocab','r') as f:
    for idx,line in enumerate(f):
        if idx < 4: continue
        term = line.split()[0]
        if term in org_vocab: continue
        if args.v_size_PM == 0:
            list_TGT_All.append(line)
            continue
        if term in PAC_topK: list_TGT_All.append(line)
    

V_TGT = list_TGT_All

print('V_TGT Size:',len(V_TGT))

def get_Union_PM(l1,l2,frac):
    ret_list = [x for x in l1]
    ret_list_keys = [x.split()[0] for x in ret_list]
    new_list = list()
    
    v_size = int(frac*len(ret_list))
    
    added = 0
    for row in l2:
        if added >= v_size: break
        
        if row.split()[0]  not in ret_list_keys:
            if row.split()[0] not in org_vocab:
                new_list.append(row)
                added +=1
    
    return ret_list+new_list

if args.frac > 0: FINAL_Vocab = get_Union_PM(V_TGT,list_PM_All,args.frac)
else: FINAL_Vocab = V_TGT
print('Final_Vocab:', len(FINAL_Vocab))

if not os.path.exists(f'{args.dataset}-PEGASUS'):
    os.makedirs(f'{args.dataset}-PEGASUS')
dump_path = f'{args.dataset}-PEGASUS/{args.v_size_PM//1000}K_{args.frac}_.txt'

with open(dump_path,'w') as f:  #put dump path as directory of your choice
    f.write(''.join(FINAL_Vocab))
f.close()
