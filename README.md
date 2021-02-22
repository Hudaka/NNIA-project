# Neural Networks Project


* In this project we will be training a model for POS using BERT and for data encoding 
 
## Table of Contents
1. **Part I: Preprocessing data in CoNLL format**
- Extract relevent information (POS tag);
- Analyse: size , classes, balanced/imbalanced, lenth of sequences
2. **Part II: Train a model**
-  Encode data using *BERT*
-  Train a model for *POS*
- Choose hyperparameters using *wandb*



Conll data preprocessing
========================

Description
-----------

This project aims to stream line the preprocessing of .conll files, in order to exact the useful columns and build a more user friendly .tsv file out of them and also provide a statistical analysis of the data processed, to provide the user with even more insight (said stats are stored in a .info file).

Use
---

The program takes two parameters: `--input` and `--outdir`, for the input conll file and the output directory respectively. The following example demonstrates how to use them (it is the same example used in run.sh):

```sh
$ python3 data_preprocess.py --input sample.conll --outdir output
```

Requirement
-----------
- Python 3.6 or higher
