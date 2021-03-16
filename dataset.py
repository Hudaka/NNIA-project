# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 19:04:06 2021

@author: sandr
"""

import config
import torch


class EntityRecognition:
    def __init__(self, sen, pos):
        #self.position = position
        self.sen = sen
        self.pos = pos
        
    def __len__(self):
        return len(self.sen)
        
    def __getitem__(self, instance):
        #position = self.position[instance]
        sen = self.sen[instance]
        pos = self.pos[instance]
        
        ids = []
        tpos = []
        
        for i,w in enumerate(sen):
            inp = config.tokenizer.encode(w, add_special_tokens=False)
        
        inplen = len(inp)
        ids.extend(inp)
        tpos.extend([pos[i]] * inplen)
        ids = ids[:config.MAX_LEN - 2]
        tpos = tpos[:config.MAX_LEN - 2]
        ids = [101] + ids + [102]
        tpos = [0] + tpos + [0]
        
        idslen = len(ids)
        
        attention_mask = [1] * idslen
        IDS = [0] * idslen
        
        padding = [0] * (config.MAX_LEN - len(ids))
        
        ids = ids + padding
        attention_mask = attention_mask + padding
        IDS = IDS + padding
        tpos = tpos + padding
        
        
        
        return {"ids": torch.tensor(ids, dtype=torch.long), 
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "IDS": torch.tensor(IDS, dtype=torch.long),
                "tpos": torch.tensor(tpos, dtype=torch.long)}