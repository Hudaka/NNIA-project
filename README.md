# NNIA PROJECT WS 2020/21

This is a project of the lecture Neural Networks: Implementation and Application from the University of Saarland. It consists of three parts: 
1. 1. [x] Preprocessing data in CoNLL format 
2. 2. [x] Train a model 
3. 3. [ ] Write a report ``` [x] introduction and methods ```

* In this project we will be training a model for POS using BERT for data encoding 
 
## Table of Contents
- [Conll data preprocessing](https://github.com/Hudaka/NN_project/blob/main/README.md#conll-data-preprocessing)
- [Description](https://github.com/Hudaka/NN_project/blob/main/README.md#description)
- [Use](https://github.com/Hudaka/NN_project/blob/main/README.md#use)
- [Requirenments](https://github.com/Hudaka/NN_project/blob/main/README.md#requirement)

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


Model Training
========================

Preparing the Data
-----

The data we used, which was made available to us and is part of the OntoNotes dataset, was concatenated from many single .gold_conll files into one .conll file 

```sh
$ type *.gold_conll > dataset_combined.conll
```

and then preprocessed by the code from the first part of the project ('data_preprocess.py') to create a .tsv file with three columns for position, word and POS-Tag. The data path for the model can be set in the config.py file. 


Named Entity Recognition Model
------------------------------

I had troubles writing my own dataset loading script even after the reading the description on the hugginface website and using their provided template. To still be able to work on the model I used the pandas module to access the data with ```pd.read_csv(path, sep='\t', encoding="utf-8", quoting=csv.QUOTE_NONE, header=None)```. (I will continue trying figuring out how '_split_generators' and '_generate_examples' work.)

Unfortunatly I had also troubles using wandb, I keep recieving an ```ImportError: cannot import name 'wandb' from 'transformers' (unknown location)``` error, eventhough I checked my version of transformers and my environment. I keep all my hyperparameters in the file config.py so. I will try fix this problem aswell.

By now I have implemented following seperate python files: 
_config.py_ = In this file we keep the hyper-parameters (max. sequence lenght, number of epochs, training size, test size and learning rate), the paths for the model (model path, training file path) and the specific BERT-based pretained model aswell as the BERT tokenizer.

_dataset.py_ = Here we have the class for the dataset where we tokenize every word input in order to be useable for BERT.. We use [101] as class token and [102] as seperator and also create the attention mask, target POS-Tags, token_type_ids and use padding on inputs so every input has the same lenght. They are returned in a dictionary format using PyTorch tensors. We will need them for our forward() method in our model.

_trainengine.py_ = This script is used for the training process. We have used a simple step-by-step training and evaluation architecture inspired by other models. I am unsure if the weight decay for my optimizer should rather be implemented after computing the gradient in ```loss.backward()``` and before the step of the optimizer in ```optimization.step()``` instead of doing it in the training.py file itself. 

_mdl.py_ = This is the model itself which contains the forward application and calculates the loss for our model encoded in the function ```loss()```, which takes our output, the expected output, the attention_mask (we don't calculate the loss for the whole sentence in BERT) and the total number of POS-Tags as arguments. 

_training.py_ = Used to execute the training via the prompt

```sh
$ python training.py
```
in the shell. Normally I would have my datascript loader used in the dataprocessing function, which encodes our POS-Tags and furthermore create a list of sentence lists from our data aswell as a list of list of encoded POS-Tags. Here I also split my data into training and test datasets. It is common to go for a 60%/20%/20% split. The values of the weight decay for the optimizer are inspired by looking at other models. **The training file path has to be set accordingly. The file needs to be in .tsv format and preprocessed before being used in the model**

**I am currently stuck in my training.py file**

Using my cpu brings the error ```data. :DefaultCPUAllocator: not enough memory: you tried to allocate 67372800 bytes. Buy new RAM!'```. So I tried using my GPU with ```device = torch.device("cuda")```. Unfortunatly, I have not figured out yet solve the error ```AssertionError: Torch not compiled with CUDA enabled```. 

Since this is my first implementation of a neural network, specifically Named Entity Recognition, I also have doubts that I thought of everything I need in order to perfom NER. Currently I am only using the POS Tags in training. My goal is it to give the model a sentence it has not seen before with an unseen entity and it correctly identifies the correct word as an entity.
