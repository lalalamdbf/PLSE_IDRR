# Prompt-based Logical Semantics Enhancement for Implicit Discourse Relation Recognition

The Code for the EMNLP 2023 main conference paper "**Prompt-based Logical Semantics Enhancement for Implicit Discourse Relation Recognition**".

## **Dependence Installation**

```
pip install -r requirements.txt
```

## Data

#### Pre-training data

We approximately collected 0.56 million explicit argument pairs from an unannotated corpus renowned as [Gigaword]( https://aclanthology.org/W12-3018/). Please note the data is only is utilized for academic research purposes.

#### Prompt-tuning data

We use PDTB 2.0 and CoNLL 2016 Shared Task to evaluate our models. If you have bought data from LDC, please put the PDTB data and CoNLL data in *src/data/pdtb2* and */src/data/conll* respectively.

## Data Preprocessing

- run the following commands, respectively:

```
unzip ./src/data/explicit_data/explicit_data.zip -d ./src/data/explicit_data
```

```
sh ./scripts/data_preprocess_pdtb2_4.sh
```

```
sh ./scripts/data_preprocess_pdtb2_11.sh
```

```
sh ./scripts/data_preprocess_conll.sh
```

## Pre-training

-  run the following command:


```
sh ./scripts/pretrain_plse.sh
```

## Prompt-tuning

- For 4-way classification on PDTB 2.0, run the following command:

```
sh ./scripts/train_pdtb_4.sh
```

- For 11-way classification on PDTB 2.0, run the following command:

```
sh ./scripts/train_pdtb_11.sh
```

- For 14-way classification on CoNLL16, run the following command:

```
sh ./scripts/train_conll_14.sh
```

## Bibliography

If you find this repo useful, please cite our paper.

```
@inproceedings{
  wang-etal-2023-PLSEIDRR,
  author = {Chenxu Wang, Ping Jian and Mu Huang},
  title = "Prompt-based Logical Semantics Enhancement for Implicit Discourse Relation Recognition",
  booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
  year = "2023",
  address = "Singapore",
  publisher = "Association for Computational Linguistics",
}
```