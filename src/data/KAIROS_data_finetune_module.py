import os 
import json 
import re 
import random 
from collections import defaultdict 
import argparse
from types import MappingProxyType 

import transformers 
from transformers import BartTokenizer
import torch 
from torch.utils.data import DataLoader 
import pytorch_lightning as pl 

from data import IEDataset, my_collate_finetune
from utils import load_ontology, check_pronoun, clean_mention
from tqdm import tqdm

MAX_CONTEXT_LENGTH=380
MAX_LENGTH=512

class KAIROSDataFinetuneModule(pl.LightningDataModule):
    '''
    Dataset processing for KAIROS. Involves chunking for long documents.
    '''
    def __init__(self, args):
        super().__init__() 
        self.hparams.update(vars(args))
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>',' <tgr>',' <tag>'])
    
    def create_context_pair(self,ex, id2ent=None, index=0):
        '''
        REUTRN:
        original_contex
        incremental_contex: contex with the hilighted arguments
        '''
        arguments = ex["event_mentions"][index]["arguments"]
        _,t1,t2 = ex["event_mentions"][index]["event_type"].split('.')
        if t2 == "Unspecified":
            evnt_type = t1.lower()
        else:
            evnt_type = t1.lower() if len(t1) < len(t2) else t2.lower()
        # for each argument we will add <tag> event_type role-type <tag>
        # for the trigger, we will add <tgr> event_type <tgr>
        tag_words = 5 * len(arguments) + 5
        max_length = MAX_CONTEXT_LENGTH - tag_words

        trigger = ex['event_mentions'][index]['trigger']
        offset = 0 
        # trigger span does not include last index 
        context_words = ex['tokens']
        center_sent = trigger['sent_idx']        
        if len(context_words) > max_length:
            cur_len = len(ex['sentences'][center_sent][0])
            
            if cur_len > max_length:
                # one sentence is very long
                trigger_start = trigger['start']
                start_idx = max(0, trigger_start- max_length//2 )
                end_idx = min(len(context_words), trigger_start + max_length //2  )
                context_words = context_words[start_idx: end_idx]
                offset = start_idx
            else:
                context_words = [tup[0] for tup in ex['sentences'][center_sent][0]]
                # take a sliding window 
                left = center_sent -1 
                right = center_sent +1 
                
                total_sents = len(ex['sentences'])
                prev_len =0 
                while cur_len >  prev_len:
                    prev_len = cur_len 
                    # try expanding the sliding window 
                    if left >= 0:
                        left_sent_tokens = [tup[0] for tup in ex['sentences'][left][0]]
                        if cur_len + len(left_sent_tokens) <= max_length:
                            context_words = left_sent_tokens + context_words
                            left -=1 
                            cur_len += len(left_sent_tokens)
                    
                    if right < total_sents:
                        right_sent_tokens = [tup[0] for tup in ex['sentences'][right][0]]
                        if cur_len + len(right_sent_tokens) <= max_length:
                            context_words = context_words + right_sent_tokens
                            right +=1 
                            cur_len += len(right_sent_tokens)
                # update trigger offset 
                offset = sum([len(ex['sentences'][idx][0]) for idx in range(left+1)])

        argument_mask = [0]*len(context_words)
        # add tags for the arguments and triggers
        add_tag = {}
        for arg in arguments:
            arg_sent = id2ent[arg['entity_id']]['sent_idx']
            arg_start = id2ent[arg['entity_id']]['start'] - offset
            arg_end = id2ent[arg['entity_id']]['end'] - offset
            if arg_start < 0 or arg_end >= len(context_words):
                continue
            if f"{arg_start}_{arg_end}" not in add_tag:
                add_tag[f"{arg_start}_{arg_end}"] = ""
            add_tag[f"{arg_start}_{arg_end}"] += 'and ' + arg["role"]
            for idx in range(arg_start,arg_end):
                argument_mask[idx] = 1
        
        arg_start = trigger['start'] - offset 
        arg_end = trigger['end'] - offset 
        if f"{arg_start}_{arg_end}" not in add_tag:
            add_tag[f"{arg_start}_{arg_end}"] = ""
        add_tag[f"{arg_start}_{arg_end}"] += 'and trigger'
        for idx in range(arg_start,arg_end):
            argument_mask[idx] = 1

        
        add_tag = [{"start":int(pos.split('_')[0]),"end":int(pos.split('_')[1]),"role":add_tag[pos][4:]} for pos in add_tag]
        add_tag = sorted(add_tag, key=lambda x: x["start"])
        # print(add_tag)

        added_context_words = []
        added_argument_mask = []
        curr = 0
        for arg in add_tag:
            pre_words = self.tokenizer.tokenize(' '.join(context_words[curr:arg['start']]), add_prefix_space=True)
            prefix = self.tokenizer.tokenize(' '.join([evnt_type, arg["role"].lower(), ":"]), add_prefix_space=True)
            arg_words = self.tokenizer.tokenize(' '.join(context_words[arg['start']:arg['end']]), add_prefix_space=True)
            added_context_words += pre_words + [' <tag>', ] + prefix + arg_words + [' <tag>', ]
            added_argument_mask += [0]*(len(pre_words)+1+len(prefix)) + [1]*len(arg_words) + [0]*1
            curr = arg['end']
        suf_words = self.tokenizer.tokenize(' '.join(context_words[curr:]), add_prefix_space=True)
        added_context_words += suf_words
        added_argument_mask += [0]*len(suf_words)

        assert(len(added_context_words) <= MAX_LENGTH) 
        assert len(added_argument_mask) == len(added_context_words)

        context_words_token = self.tokenizer.tokenize(' '.join(context_words), add_prefix_space=True)
        argument_mask_token = [0]*len(context_words_token)

        ptr = 0
        for idx,token in enumerate(context_words_token):
            argument_mask_token[idx] = argument_mask[ptr]
            if idx+1<len(context_words_token) and context_words_token[idx+1][0] == "Ġ":
                ptr += 1                


        # print("_"*40)
        # print(context_words_token)
        tmp = "".join(context_words_token)
        print(tmp.replace("Ġ"," "))
        # print(added_context_words)
        tmp = "".join(added_context_words)
        print(tmp.replace("Ġ"," "))
        assert 1==0
        # print(argument_mask_token)
        # print(added_argument_mask)
        assert sum(argument_mask_token) == sum(added_argument_mask)
        
        return context_words_token, added_context_words, argument_mask_token, added_argument_mask

            
    def prepare_data(self):
        data_dir = 'preprocessed_finetune_{}'.format(self.hparams.dataset)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            ontology_dict = load_ontology(self.hparams.dataset) 
            max_tokens = 0

            

            for split,f in [('train',self.hparams.train_file), ('val',self.hparams.val_file), ('test',self.hparams.test_file)]:
                coref_split = 'dev' if split=='val' else split 
                coref_reader = open(os.path.join(self.hparams.coref_dir, '{}.jsonlines'.format(coref_split)))
                with open(f,'r') as reader,  open(os.path.join(data_dir,'{}.jsonl'.format(split)), 'w') as writer:
                    for line, coref_line in tqdm(zip(reader, coref_reader)):
                        ex = json.loads(line.strip())
                        corefs = json.loads(coref_line.strip())
                        assert(ex['doc_id'] == corefs['doc_key'])
                        # mapping from entity id to information mention
                        ent2info = {} 
                        for cidx, cluster in enumerate(corefs['clusters']):
                            for eid in cluster:
                                ent2info[eid] = corefs['informative_mentions'][cidx]

                        # mapping from entity id to entity info
                        id2ent = {}
                        for entity in ex["entity_mentions"]:
                            id2ent[entity["id"]] = entity
                            
                        for i in range(len(ex['event_mentions'])):
                            if split=='train' and len(ex['event_mentions'][i]['arguments']) ==0:
                                # skip mentions with no arguments 
                                continue 
                            # print(ex['event_mentions'][i]['arguments'])
                            # print(id2ent[ex['event_mentions'][i]['arguments'][0]['entity_id']])
                            # assert 1==0
                            evt_type = ex['event_mentions'][i]['event_type']

                            if evt_type not in ontology_dict: # should be a rare event type 
                                continue 
                                
                            original_context, incremental_context, original_mask, incremental_mask = self.create_context_pair(ex, id2ent=id2ent, index=i)

                            max_tokens = max(len(incremental_context) + 2, max_tokens)
                            assert max_tokens < MAX_LENGTH

                            original_mask_pad = [0] + original_mask + [0]*(MAX_LENGTH - 1 - len(original_mask))
                            incremental_mask_pad = [0] + incremental_mask + [0]*(MAX_LENGTH - 1 - len(incremental_mask))

                            original_tokens = self.tokenizer.encode_plus(original_context, 
                            add_special_tokens=True,
                            add_prefix_space=True, 
                            max_length=MAX_LENGTH,
                            truncation=True,
                            padding='max_length')

                            incremental_tokens = self.tokenizer.encode_plus(incremental_context, 
                            add_special_tokens=True,
                            add_prefix_space=True, 
                            max_length=MAX_LENGTH,
                            truncation=True,
                            padding='max_length')
                            
                            processed_ex = {
                                'event_idx': i, 
                                'doc_key': ex['doc_id'], 
                                'input_token_ids_1':original_tokens['input_ids'],
                                'input_attn_mask_1': original_tokens['attention_mask'],
                                'argument_mask_1': original_mask_pad,
                                'input_token_ids_2':incremental_tokens['input_ids'],
                                'input_attn_mask_2': incremental_tokens['attention_mask'],
                                'argument_mask_2': incremental_mask_pad,
                            }
                            writer.write(json.dumps(processed_ex) + '\n')
            

            print('longest context:{}'.format(max_tokens))
    
    def train_dataloader(self):
        dataset = IEDataset('preprocessed_finetune_{}/train.jsonl'.format(self.hparams.dataset))
        
        dataloader = DataLoader(dataset, 
            pin_memory=True, num_workers=2, 
            collate_fn=my_collate_finetune,
            batch_size=self.hparams.train_batch_size, 
            shuffle=True)
        return dataloader 

    
    def val_dataloader(self):
        dataset = IEDataset('preprocessed_finetune_{}/val.jsonl'.format(self.hparams.dataset))
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2, 
            collate_fn=my_collate_finetune,
            batch_size=self.hparams.eval_batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        dataset = IEDataset('preprocessed_finetune_{}/test.jsonl'.format(self.hparams.dataset))
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2, 
            collate_fn=my_collate_finetune, 
            batch_size=self.hparams.eval_batch_size, shuffle=False)

        return dataloader


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--train-file',type=str,default='data/wikievents/train.jsonl')
    parser.add_argument('--val-file', type=str, default='data/wikievents/dev.jsonl')
    parser.add_argument('--test-file', type=str, default='data/wikievents/test.jsonl')
    parser.add_argument('--coref-dir', type=str, default='data/wikievents/coref')
    parser.add_argument('--use_info', action='store_true', default=True, help='use informative mentions instead of the nearest mention.')
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--dataset', type=str, default='KAIROS')
    parser.add_argument('--mark-trigger', action='store_true', default=True)
    args = parser.parse_args() 

    dm = KAIROSDataFinetuneModule(args)
    dm.prepare_data() 

    # training dataloader 
    dataloader = dm.train_dataloader() 

    for idx, batch in enumerate(dataloader):
        print(batch)
        break 

    # val dataloader 