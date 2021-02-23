# NNIA PROJECT WS 2020/21

This is a project of the lecture Neural Networks: Implementation and Application from the University of Saarland. It consists of three parts: 
1. 1. [x] Preprocessing data in CoNLL format 
2. 2. [ ] Train a model 
3. 3. [ ] Write a report

* In this project we will be training a model for POS using BERT for data encoding 
 
## Table of Contents
- [Conll data preprocessing](https://github.com/Hudaka/NN_project/blob/main/README.md#conll-data-preprocessing)
- [Description](https://github.com/Hudaka/NN_project/blob/main/README.md#description)
- [Use](https://github.com/Hudaka/NN_project/blob/main/README.md#use)




# Conll data preprocessing
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
