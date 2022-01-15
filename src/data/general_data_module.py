import os 
import json 
import re 
import random 
from collections import defaultdict 
import argparse 

import transformers 
from transformers import BartTokenizer
import torch 
from torch.utils.data import DataLoader 
import pytorch_lightning as pl
from transformers.file_utils import torch_required
from transformers.utils.dummy_tokenizers_objects import DPRQuestionEncoderTokenizerFast 

from .data import IEDataset, my_collate_event_aware
from ..genie.utils import load_ontology, check_pronoun, clean_mention


class GeneralDataModule(pl.LightningDataModule):
    '''
    Dataset processing for KAIROS. Involves chunking for long documents.
    '''
    def __init__(self, args, train_file = None, val_file = None, test_file = None):
        super().__init__() 
        self.hparams = args
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>',' <tgr>', ' <tag>', ' </tag>'])
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
    
    def train_dataloader(self):
        dataset = IEDataset(self.train_file)

        dataloader = DataLoader(dataset, 
            pin_memory=True, num_workers=2, 
            collate_fn=my_collate_event_aware,
            batch_size=self.hparams.train_batch_size, 
            shuffle=True)
        return dataloader 

    
    def val_dataloader(self):
        dataset = IEDataset(self.val_file)
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2, 
            collate_fn=my_collate_event_aware,
            batch_size=self.hparams.eval_batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        dataset = IEDataset(self.test_file)
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2, 
            collate_fn=my_collate_event_aware, 
            batch_size=self.hparams.eval_batch_size, shuffle=False)

        return dataloader