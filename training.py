#import wandb
from transformers import get_linear_schedule_with_warmup
from sklearn import preprocessing 
import config
import csv
import pandas as pd
from transformers import AdamW
from sklearn import model_selection

import trainengine
import dataset 
import torch
from mdl import EntityRecognitionModel
import numpy as np


def dataprocessing(path):
    #file = open("datasetsv.tsv", encoding="utf-8")#data
    #tsv = csv.reader(file, delimiter="\t")
    tsv_read = pd.read_csv(path, sep='\t', encoding="utf-8", quoting=csv.QUOTE_NONE, header=None)
    #print(tsv_read)
    tsv_read.columns = ['position', 'word', 'pos']
    sentence_list = []
    pos_list = []
    
    helplist_s = list()
    helplist_p = list()
    
    pos_encoder = preprocessing.LabelEncoder()
    tsv_read.loc[:, 'pos'] = pos_encoder.fit_transform(tsv_read['pos'].transform(str))
    
    for ind in tsv_read.index:
        if tsv_read['position'][ind] != '*':
            helplist_s.append(tsv_read['word'][ind])
            helplist_p.append(tsv_read['pos'][ind])
        else:
            sentence_list.append(helplist_s)
            pos_list.append(helplist_p)
            helplist_s = []
            helplist_p = []
    return sentence_list, pos_list, pos_encoder
            
            
if __name__ == "__main__":
    sentence_list, pos_list, pos_encoder  = dataprocessing(config.TRAINING_FILE)
    
    hyperdata = {"pos_encoder": pos_encoder}
    
    number_of_pos = len(list(pos_encoder.classes_))
    
    
    (TRAIN_S, TEST_S, TRAIN_P, TEST_P) = model_selection.train_test_split(sentence_list, pos_list, random_state=20, test_size=0.2)


    train_dataset = dataset.EntityRecognition(
        sen=TRAIN_S, pos=TRAIN_P
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.EntityRecognition(
        sen=TEST_S, pos=TEST_P
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cpu")
    model = EntityRecognitionModel(postotal=number_of_pos)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(TRAIN_S) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = trainengine.train(train_data_loader, model, optimizer, device, scheduler)
        test_loss = trainengine.eval_fn(valid_data_loader, model, device)
        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss


            
        