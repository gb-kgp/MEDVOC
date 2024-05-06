#!/bin/zsh

python GenerateSubwordsBERT.py \
       -dataset EBM \
       -path_pubmed /Users/gunjanbalde/Documents/SR-NG-MedVoc/SR-NG-MedVoc/MedicalDataset/CSVs_PubMed/Train_PubMed.csv  \
       -path_data /Users/gunjanbalde/Documents/SR-NG-MedVoc/SR-NG-MedVoc/MedicalDataset/CSVs_EBM/Filtered_Train_BART.csv \
       -csv_path ./TokenSplitDistribution-EBM_BERT.csv \
       -reference_column summary_text

python GenerateSubwordsBERT.py \
       -dataset BioASQ \
       -path_pubmed /Users/gunjanbalde/Documents/SR-NG-MedVoc/SR-NG-MedVoc/MedicalDataset/CSVs_PubMed/Train_PubMed.csv  \
       -path_data /Users/gunjanbalde/Documents/SR-NG-MedVoc/SR-NG-MedVoc/MedicalDataset/BioASQ_Bert/BioASQ_CSVs/Train.csv \
       -csv_path ./TokenSplitDistribution-BioASQ_BERT.csv \
       -reference_column target_text


python GenerateSubwordsBERT.py \
       -dataset MeQSum \
       -path_pubmed /Users/gunjanbalde/Documents/SR-NG-MedVoc/SR-NG-MedVoc/MedicalDataset/CSVs_PubMed/Train_PubMed.csv  \
       -path_data /Users/gunjanbalde/Documents/SR-NG-MedVoc/SR-NG-MedVoc/MedicalDataset/MeQSum/Train.csv \
       -csv_path ./TokenSplitDistribution-MeQSum_BERT.csv \
       -reference_column Summary

python GenerateSubwordsBERT.py \
         -dataset CHQ \
         -path_pubmed /Users/gunjanbalde/Documents/SR-NG-MedVoc/SR-NG-MedVoc/MedicalDataset/CSVs_PubMed/Train_PubMed.csv  \
         -path_data /Users/gunjanbalde/Documents/SR-NG-MedVoc/SR-NG-MedVoc/MedicalDataset/chq-summ-sd-rs/CHQ-CSVs/Train.csv \
         -csv_path ./TokenSplitDistribution-CHQ_BERT.csv
         -reference_column target_text