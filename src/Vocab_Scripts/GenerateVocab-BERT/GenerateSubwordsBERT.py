from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import pandas as pd
import argparse
import os
import tqdm

if not os.path.exists('VocabFiles_BERT'): os.mkdir('VocabFiles_BERT')

parser = argparse.ArgumentParser()
parser.add_argument('-dataset',type=str,required=True)
parser.add_argument('-csv_path',type=str,required=True)

args = parser.parse_args()

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
base_vocabs = tokenizer_bert.get_vocab()

if not os.path.exists(args.dataset):
    os.mkdir(args.dataset)

df_PM = pd.read_csv(args.csv_path)
df_consider = df_PM[df_PM['Consider']==1]
df_PM = df_consider[df_consider['TokenFrom']=='PubMed']

df_TGT = pd.read_csv(args.csv_path)
df_consider = df_TGT[df_TGT['Consider']==1]
df_TGT = df_consider[df_consider['TokenFrom']==args.dataset]

list_PM_GT4, list_PM_LT4 = list(), list()

list_PM_All = list()
for idx in tqdm.tqdm(range(df_PM.shape[0])):
    list_PM_All.extend([df_PM.iloc[idx,0]]*df_PM.iloc[idx,2])

with open('VocabFiles_BERT/PM_BertSumAbs_All','w') as f:
     f.write('\n'.join(list_PM_All))
f.close()
            
list_TGT_All = list()

for idx in tqdm.tqdm(range(df_TGT.shape[0])): list_TGT_All.extend([df_TGT.iloc[idx,0]]*df_TGT.iloc[idx,2])

with open(f'VocabFiles_BERT/{args.dataset}_BertSumAbs_All','w') as f:
    f.write('\n'.join(list_TGT_All))
f.close()
    


for v_size in [15000]: #[5000, 10000, 15000, 20000, 25000, 30000]:
    for alpha in [0.25]:   #[0,0.25,0.5,0.75,1]:
        tokenizer_rs = BertWordPieceTokenizer()
        tokenizer_rs.train(f'VocabFiles_BERT/{args.dataset}_BertSumAbs_All',show_progress=True)
        RS_vocab = [x[0] for x in sorted(tokenizer_rs.get_vocab().items(),key=lambda x:x[1])]
        RS_abs = [y for y in RS_vocab if y not in base_vocabs]

        tokenizer_pubmed = BertWordPieceTokenizer()
        tokenizer_pubmed.train('VocabFiles_BERT/PM_BertSumAbs_All',vocab_size=v_size,show_progress=False)
        vocab_pubmed = tokenizer_pubmed.get_vocab()
        vocab_pubmed = [y for y in vocab_pubmed if y not in base_vocabs]

        RS_abs_imp = [y for y in RS_abs if y in vocab_pubmed]
        
        v_tgt = RS_abs_imp
        v_pac = vocab_pubmed[:int((alpha+1)*len(v_tgt))+1]

        print(len(v_tgt),len(v_pac),len(set(v_tgt).intersection(set(v_pac)))/len(v_tgt),1-len(RS_abs_imp)/len(RS_abs))


        
        # vocab_pubmed = [y for y in vocab_pubmed if y not in final_vocab]
        # final_vocab_new = final_vocab + vocab_pubmed[:int(alpha*len(final_vocab))]
        
        # # with open(f'{args.dataset}/vocab_DRIFT_{v_size//1000}K_{alpha}_NoK-A.txt','w') as f:
        # #     # f.write('\n'.join(base_vocabs)+'\n'+'\n'.join(final_vocab_new))
        # #     f.write('\n'.join(base_vocabs)+'\n'+'\n'.join(RS_abs))
        # # f.close()
        
        # print(f'Done for {v_size//1000}K_{alpha}: {len(RS_abs)}')


'''
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

from transformers import BertTokenizer
import glob
from collections import defaultdict

dict_scores = defaultdict(lambda : defaultdict(dict))
for fname in sorted(glob.glob(f'{args.dataset}/*NoK-A*'),key = lambda x: [int(x.split('/')[-1].split('_')[-3][:-1]),float(x.split('/')[-1].split('_')[-2])]):
    domain_tok = BertTokenizer.from_pretrained(fname)
    sum_num = 0.
    sum_den = 0.
    
    for idx,term in enumerate(terms_EBM):
        sum_num += len(domain_tok.tokenize(term))*freq_ebm[idx]
        sum_den += freq_ebm[idx]

    print(fname, sum_num/sum_den, len(domain_tok))
    key = fname.split('/')[-1].split('_')
    dict_scores[key[-3]][key[-2]] = [round(sum_num/sum_den,2),len(domain_tok)]

for k1 in dict_scores:
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
'''
