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

from data import IEDataset, my_collate_event_aware
from utils import load_ontology, check_pronoun, clean_mention

MAX_CONTEXT_LENGTH=350 # measured in words
WORDS_PER_EVENT=10 
MAX_LENGTH=512
MAX_TGT_LENGTH=70

class KAIROSDataSharedArgModule(pl.LightningDataModule):
    '''
    Dataset processing for KAIROS. Involves chunking for long documents.
    '''
    def __init__(self, args):
        super().__init__() 
        self.hparams = args
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>',' <tgr>', ' <tag>', ' </tag>'])
    
    def simple_type(self, type_name):
        _,t1,t2 = type_name.split('.')
        if t2 == "Unspecified":
            return t1.lower()
        if len(t1) < len(t2):
            return t1.lower()
        return t2.lower()
            
    def prepare_data(self):
        data_dir = 'preprocessed_shared_arg_{}'.format(self.hparams.dataset)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            ontology_dict = load_ontology(self.hparams.dataset) 
            max_tokens = 0
            max_tgt =0 
            cnt = 0
            arg_cnt = defaultdict(list)
            for split,f in [('train',self.hparams.train_file), ('val',self.hparams.val_file), ('test',self.hparams.test_file)]:
                coref_split = 'dev' if split=='val' else split 
                coref_reader = open(os.path.join(self.hparams.coref_dir, '{}.jsonlines'.format(coref_split)))
                with open(f,'r') as reader,  open(os.path.join(data_dir,'{}.jsonl'.format(split)), 'w') as writer:
                    for line, coref_line in zip(reader, coref_reader):
                        ex = json.loads(line.strip())
                        corefs = json.loads(coref_line.strip())
                        assert(ex['doc_id'] == corefs['doc_key'])
                        
                        # mapping from entity id to entity info
                        id2entity = {}
                        for entity in ex["entity_mentions"]:
                            id2entity[entity["id"]] = entity
                        
                        event_arg = {}
                        for i in range(len(ex['event_mentions'])):
                            if len(ex['event_mentions'][i]['arguments']) > 0:
                                event_arg[i] = set(x['entity_id'] for x in ex['event_mentions'][i]['arguments'])
                        events = event_arg.keys()
                        # events = sorted(events, key = lambda x:event_range[x]['start'])


                        for i in range(len(ex['event_mentions'])):
                            if len(ex['event_mentions'][i]['arguments']) ==0:
                                # skip mentions with no arguments 
                                continue 
                            evt_type = ex['event_mentions'][i]['event_type']
                            if evt_type not in ontology_dict: # should be a rare event type 
                                continue 

                            close_events = list(filter(lambda x: len(event_arg[i].intersection(event_arg[x])) and x!=i, events)) # events sharing the same arguments with the current event
                            
                            arg_1 = {x['entity_id']:x for x in ex['event_mentions'][i]['arguments']}
                            for eid in close_events:
                                arguments = ex['event_mentions'][eid]['arguments']
                                for arg in arguments:
                                    if arg['entity_id'] in arg_1:
                                        tmp_list = [arg["role"],arg_1[arg['entity_id']]["role"]]
                                        tmp_list = sorted(tmp_list)
                                        arg_cnt["-".join(tmp_list)].append(1)
            
            arg_cnt = list(arg_cnt.items())
            arg_cnt = sorted(arg_cnt,key=lambda x:sum(x[1]),reverse=True)
            print(arg_cnt)
            for i,pair in enumerate(arg_cnt):
                print(f'{i} - {pair[0]}: {sum(pair[1])}')
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train-file',type=str,default='data/wikievents/train.jsonl')
    parser.add_argument('--val-file', type=str, default='data/wikievents/dev.jsonl')
    parser.add_argument('--test-file', type=str, default='data/wikievents/test.jsonl')
    parser.add_argument('--coref-dir', type=str, default='data/wikievents/coref')
    parser.add_argument('--use_info', action='store_true', default=False, help='use informative mentions instead of the nearest mention.')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='KAIROS')
    parser.add_argument('--mark-trigger', action='store_true', default=True)
    args = parser.parse_args() 

    dm = KAIROSDataSharedArgModule(args=args)
    dm.prepare_data() 
