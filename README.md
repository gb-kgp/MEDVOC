# Codebase for MEDVOC
 The official codebase for our IJCAI 2024 submission titled "MEDdVOC: Vocabulary Adaptation for Fine-tuning Pre-trained Language Models on Medical Text Summarization". We also provide the technical appendix for the submission. Preprint for the submission can be found [here](https://arxiv.org/abs/2405.04163)

 1.  ```src``` folder contains codebase for vocabulary adaption algorithm along with the modified codes to handle the models with updated vocabulary.

 2. ```data``` folder contains details on how you can acquire data that we used in this work.

Before you begin, please create two seperate python envs one for transformers (using requirements_transformers.txt) as follows:
```
conda create -n env_Transformers python=3.8
pip install -r requirements_transformers.txt
```

and one for using QuickUMLS (using requirements_UMLS.txt) follwoing the setup as described in [QuickUMLS](https://github.com/Georgetown-IR-Lab/QuickUMLS) repo.


If you plan to use this work, please use the following bibtex entry:

```
@inproceedings{ijcai2024p683,
  title     = {MEDVOC: Vocabulary Adaptation for Fine-tuning Pre-trained Language Models on Medical Text Summarization},
  author    = {Balde, Gunjan and Roy, Soumyadeep and Mondal, Mainack and Ganguly, Niloy},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {6180--6188},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/683},
  url       = {https://doi.org/10.24963/ijcai.2024/683},
}
```
