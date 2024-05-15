#!/bin/zsh

python GenerateSubwordsBERT.py \
       -dataset EBM \
       -path_pubmed ../../../data/Dataset-CSVs/PAC/Train.csv \
       -path_data ../../../data/Dataset-CSVs/EBM/Train.csv \
       -csv_path ./TokenSplitDistribution-EBM_BERT.csv \
       -reference_column summary_text