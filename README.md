# NNIA PROJECT WS 2020/21

This is a project of the lecture Neural Networks: Implementation and Application from the University of Saarland. It consists of three parts: 
1. 1. [x] Preprocessing data in CoNLL format 
2. 2. [x] Train a model 
3. 3. [x] Write a report

* In this project we will be training a model for POS using BERT for data encoding 
 
## Table of Contents
- [Conll data preprocessing](https://github.com/Hudaka/NN_project/blob/main/README.md#conll-data-preprocessing)
- [Description](https://github.com/Hudaka/NN_project/blob/main/README.md#description)
- [Use](https://github.com/Hudaka/NN_project/blob/main/README.md#use)
- [Requirenments](https://github.com/Hudaka/NN_project/blob/main/README.md#requirement)
- [Combine multiple files](https://github.com/Hudaka/NNIA-project/blob/main/README.md#combine-multiple-files)

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

Requirenment
-----------
- Python 3.6 or higher


Combine multiple files
-----------

To concatenate multiple CoNLL files use the following command in the terminal. It will concatenate 4 CoNLL files into one file named sample.conll.

 

For Windows:

 

```sh

$ type file1.conll file2.conll file3.conll file4.conll > sample.conll

```

 

For Linux or MAC:

 

```sh

$ cat file1.conll file2.conll file3.conll file4.conll > sample.conll

```

