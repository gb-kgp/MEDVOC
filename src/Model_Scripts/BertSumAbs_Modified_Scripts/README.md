We add a parameter ```-vocab_file``` to account for the newly added vocabulary. Our base PLM here is BertSumAbs (bert-base-uncased).

Let's say you want to train the model for BioASQ, and you have the preprocessed data available as discussed in the original PreSummCodebase, simply run the following:
```
python src/train.py  -task abs -mode train -bert_data_path bert_PubMed_NewOOV_BioASQ_15K_0.25/pubmed \
                     -dec_dropout 0.2 -model_path BertSumAbs_TAP_NewOOV_BioASQ_15K_0.25_0.002_0.01_10 \ 
                     -sep_optim true -lr_bert 0.002 -lr_dec 0.01 -save_checkpoint_steps 2500 \
                     -batch_size  145 -train_steps 200000 -report_every 50 -accum_count 10 \
                     -use_bert_emb true -use_interval true -warmup_steps_bert 20000 \
                     -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 2  \
                     -log_file logs/abs_TAP_NewOOV_15K_0.25 -max_tgt_len 145 \
                     -vocab_file  BertSumAbs_BioASQ_15K_0.25.pkl

```

The detials for each of the parameters can be found in the official codebase: [PreSumm](https://github.com/nlpyang/PreSumm).