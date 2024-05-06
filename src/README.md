```Model_Scripts``` contains the modified scripts for BertSumAbs and BART/PEGASUS model. We provide shell scripts in the relevant folder to run the scripts.

```Vocab_Scripts``` contains the codebase for obtaining the modified vocabulary.

Ideally you should follow this sequential steps:
```
1. Organize the data into train/test/val splits as csvs/jsons (for BART and PEGASUS). For BertSumAbs please follow the preprocessing steps as discussed in their codebase.

2. Run the scripts in following order:
  2a. First run the shell scripts in GenerateCandidateSubwords to get the tokens fit for subwords consideration.
  2b. Then run the python notebook in FilterMedicalWords to obtain medically relevant OOVs.
  2c. Then run the scripts in each of the Generate-* folders to obtain vocabularies specific to each of the models. For PEGASUS additionaly run the .ipynb file with instructions specified in each cell.
  2d. Select the optimal configuration based on fragment score as described in the paper.

3. Run the shell script in Model_Scripts to obtain the final trained models ready for inference.
```

Please contact corresponding author at balde.gunjan0812@kgpian.iitkgp.ac.in for direct access of models. We are working on making the checkpoints accessible on huggingface.

