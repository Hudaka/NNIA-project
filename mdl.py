import config, torch
import torch.nn as nn


def loss(out, exp, attention_mask, totallabels):
    fn = nn.CrossEntropyLoss()
    loss_no_padding = attention_mask.view(-1) == 1
    activeLoss = out.view(-1, totallabels)
    lbs = torch.where(loss_no_padding, exp.view(-1), torch.tensor(fn.ignore_index).type_as(exp))
    finalLoss = fn(activeLoss, lbs)
    return finalLoss


class EntityRecognitionModel(nn.Module):
    def __init__(self, postotal):
        super(EntityRecognitionModel, self).__init__()
        self.postotal = postotal
        self.bert = config.model
        self.dropout = nn.Dropout(0.1)
        self.outpos = nn.Linear(768, self.postotal)
       
        
        
        def forward(self, ids, attention_mask, IDS, tpos):
            x, y = self.bert(ids, attention_mask=attention_mask, token_type_ids = IDS )
            
            
            
            POS = self.outpos(self.dropout(x))
            
            lossPos = loss(POS, tpos, attention_mask, self.postotal)
            
            return POS, lossPos
            
            