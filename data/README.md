# Datasets
We use 4 datasets in our study that are evaluated across three PLMs. Here we provide the publically available datasets -- EBM and MeQSum. We provide two kinds of data folders.
 1. BertSumAbs-Data: In this, there is a folder specific to each target task dataset. Inside each folder, you will find two folders: (i) Raw_Files: a folder containing each data point as text files, and (ii) urls: containing the train-valid-test split of the dataset. Use the script in ```src/Model_Scripts/BertSumAbs_Modified_Scripts/src/preprocess.py``` with appropriate cmd-line arguments to generate the data in format specific to BertSumAbs.

 2. BART-PEGASUS-CSVs: This contains train-valid-test splits as csvs which can be directly given as input to run_summarization.py script ```src/Model_Scripts/Transformers_Modified_Script/run_summarization.py``` for BART and PEGASUS models. The source document column is ```input_text``` and the reference summary column is ```target_text```.

 Please refer to the codebase of the specific models for further details.
