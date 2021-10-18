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

from .data import IEDataset, my_collate
from .utils import load_ontology, check_pronoun, clean_mention

MAX_CONTEXT_LENGTH=400 # measured in words 
MAX_LENGTH=512
MAX_TGT_LENGTH=70

class KAIROSDataMultiEventModule(pl.LightningDataModule):
    '''
    Dataset processing for KAIROS. Involves chunking for long documents.
    '''
    def __init__(self, args):
        super().__init__() 
        self.hparams.update(vars(args))
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>',' <tgr>', ' <tag>'])

    
    def create_gold_gen(self, ex, ontology_dict,mark_trigger=True, index=0, ent2info=None, use_info=False):
        '''
        If there are multiple events per example, use index parameter.

        Input: <s> Template with special <arg> placeholders </s> </s> Passage </s>
        Output: <s> Template with arguments and <arg> when no argument is found. 
        '''
        if use_info and ent2info==None:
            raise ValueError('entity to informative mention mapping required.')

        evt_type = ex['event_mentions'][index]['event_type']

        
        template = ontology_dict[evt_type]['template']
        input_template = re.sub(r'<arg\d>', '<arg>', template) 


        space_tokenized_input_template = input_template.split()
        tokenized_input_template = [] 
        for w in space_tokenized_input_template:
            tokenized_input_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))
        
        role2arg = defaultdict(list)

        for argument in ex['event_mentions'][index]['arguments']:
            role2arg[argument['role']].append(argument)

        role2arg = dict(role2arg)

        # create output template 
        arg_idx2text = defaultdict(list)
        for role in role2arg.keys():
            if role not in ontology_dict[evt_type]:
                # annotation error 
                continue 
            for i, argument in enumerate(role2arg[role]):
                use_arg = True 
                if use_info:
                    ent_id = argument['entity_id']
                    if ent_id in ent2info:
                        arg_text = clean_mention(ent2info[ent_id])
                        if check_pronoun(arg_text):
                            # skipping this argument 
                            use_arg = False 
                        # if arg_text != argument['text']:
                            # print('Original mention:{}, Informative mention:{}'.format(argument['text'], arg_text))
                    else:
                        arg_text = argument['text']
                else:
                    arg_text = argument['text']
                
                # assign the argument index 
                if i < len(ontology_dict[evt_type][role]):
                    # enough slots to fill in 
                    arg_idx = ontology_dict[evt_type][role][i]
                    
                else:
                    # multiple participants for the same role 
                    arg_idx = ontology_dict[evt_type][role][-1]

                if use_arg:
                    arg_idx2text[arg_idx].append(arg_text)
                
        for arg_idx, text_list in arg_idx2text.items():
            text = ' and '.join(text_list)
            template = re.sub('<{}>'.format(arg_idx), text, template)

            

        trigger = ex['event_mentions'][index]['trigger']
        offset = 0 
        # trigger span does not include last index 
        context_words = ex['tokens']
        center_sent = trigger['sent_idx']
        if len(context_words) > MAX_CONTEXT_LENGTH:
            cur_len = len(ex['sentences'][center_sent][0])
            context_words = [tup[0] for tup in ex['sentences'][center_sent][0]]
            if cur_len > MAX_CONTEXT_LENGTH:
                # one sentence is very long
                trigger_start = trigger['start']
                start_idx = max(0, trigger_start- MAX_CONTEXT_LENGTH//2 )
                end_idx = min(len(context_words), trigger_start + MAX_CONTEXT_LENGTH //2  )
                context_words = context_words[start_idx: end_idx]
                offset = start_idx 

            else:
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
                        if cur_len + len(left_sent_tokens) <= MAX_CONTEXT_LENGTH:
                            context_words = left_sent_tokens + context_words
                            left -=1 
                            cur_len += len(left_sent_tokens)
                    
                    if right < total_sents:
                        right_sent_tokens = [tup[0] for tup in ex['sentences'][right][0]]
                        if cur_len + len(right_sent_tokens) <= MAX_CONTEXT_LENGTH:
                            context_words = context_words + right_sent_tokens
                            right +=1 
                            cur_len += len(right_sent_tokens)
                # update trigger offset 
                offset = sum([len(ex['sentences'][idx][0]) for idx in range(left+1)])
        
            
        assert(len(context_words) <= MAX_CONTEXT_LENGTH) 

        trigger['start'] = trigger['start'] - offset 
        trigger['end'] = trigger['end'] - offset 
        if mark_trigger:
            prefix = self.tokenizer.tokenize(' '.join(context_words[:trigger['start']]), add_prefix_space=True) 
            tgt = self.tokenizer.tokenize(' '.join(context_words[trigger['start']: trigger['end']]), add_prefix_space=True)
            
            suffix = self.tokenizer.tokenize(' '.join(context_words[trigger['end']:]), add_prefix_space=True)
            context = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix 
        else:
            context = self.tokenizer.tokenize(' '.join(context_words), add_prefix_space=True)

        output_template = re.sub(r'<arg\d>','<arg>', template ) 
        space_tokenized_template = output_template.split()
        tokenized_template = [] 
        for w in space_tokenized_template:
            tokenized_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))
        
        return tokenized_input_template, tokenized_template, context

    

            
    def prepare_data(self):
        cnt = 0
        two_cnt = 0
        one_cnt = 0
        data_dir = 'preprocessed_multievent_{}'.format(self.hparams.dataset)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            ontology_dict = load_ontology(self.hparams.dataset) 
            max_tokens = 0
            max_tgt =0 

            

            for split,f in [('train',self.hparams.train_file), ('val',self.hparams.val_file), ('test',self.hparams.test_file)]:
                coref_split = 'dev' if split=='val' else split 
                coref_reader = open(os.path.join(self.hparams.coref_dir, '{}.jsonlines'.format(coref_split)))
                with open(f,'r') as reader,  open(os.path.join(data_dir,'{}.jsonl'.format(split)), 'w') as writer:
                    for line, coref_line in zip(reader, coref_reader):
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

                        evnt_range = {}
                        for i in range(len(ex['event_mentions'])):
                            if len(ex['event_mentions'][i]['arguments']) > 0:
                                args = [id2ent[x['entity_id']] for x in ex['event_mentions'][i]['arguments']]
                                arg_start = [x['start'] for x in args]
                                arg_end = [x['end'] for x in args]
                                start = min(arg_start)
                                end = min(arg_start)
                                evnt_range[i] = {'start':start,'end':end}
                        

                        for i in range(len(ex['event_mentions'])):
                            if split=='train' and len(ex['event_mentions'][i]['arguments']) ==0:
                                # skip mentions with no arguments 
                                continue 
                            evt_type = ex['event_mentions'][i]['event_type']

                            if evt_type not in ontology_dict: # should be a rare event type 
                                continue 

                            
                            input_template, output_template, context= self.create_gold_gen(ex, ontology_dict, self.hparams.mark_trigger, 
                                index=i, ent2info=ent2info, use_info=self.hparams.use_info)
                            if i in evnt_range:
                                evnt_range[i] = {
                                    'start':evnt_range[i]['start'],
                                    'end':evnt_range[i]['end'],
                                    'input_template':input_template,
                                    'output_template':output_template,
                                    'context':context
                                }
                            
                            max_tokens = max(len(context) + len(input_template) +2, max_tokens)
                                # print(len(context) + len(input_template) +2 ) 
                            max_tgt = max(len(output_template) +1 , max_tgt)
                            assert(len(output_template) < MAX_TGT_LENGTH)
                            input_tokens = self.tokenizer.encode_plus(input_template, context, 
                                    add_special_tokens=True,
                                    add_prefix_space=True,
                                    max_length=MAX_LENGTH,
                                    truncation='only_second',
                                    padding='max_length')
                            tgt_tokens = self.tokenizer.encode_plus(output_template, 
                            add_special_tokens=True,
                            add_prefix_space=True, 
                            max_length=MAX_TGT_LENGTH,
                            truncation=True,
                            padding='max_length')

                            processed_ex = {
                                'event_idx': i, 
                                'doc_key': ex['doc_id'], 
                                'input_token_ids':input_tokens['input_ids'],
                                'input_attn_mask': input_tokens['attention_mask'],
                                'tgt_token_ids': tgt_tokens['input_ids'],
                                'tgt_attn_mask': tgt_tokens['attention_mask'],
                            }
                            writer.write(json.dumps(processed_ex) + '\n')
                            one_cnt += 1

                        evnt_ids = evnt_range.keys()
                        evnt_ids = sorted(evnt_ids,key=lambda x:evnt_range[x]["start"])
                        for idx in range(len(evnt_ids) - 1):
                            id1 = evnt_ids[idx]
                            id2 = evnt_ids[idx+1]
                            if evnt_range[id1]['start'] == evnt_range[id2]['start'] or evnt_range[id1]['end'] >= evnt_range[id2]['start']:
                                trg1_start = evnt_range[id1]['context'].index(' <tgr>')
                                trg1_end = evnt_range[id1]['context'].index(' <tgr>',trg1_start + 1)
                                trg2_start = evnt_range[id2]['context'].index(' <tgr>')
                                trg2_end = evnt_range[id2]['context'].index(' <tgr>',trg2_start + 1)
                                if trg1_start == trg2_start:
                                    continue
                                if trg2_start < trg1_start:
                                    context = evnt_range[id2]['context'][:trg2_end+1] + evnt_range[id1]['context'][trg2_end - 1:]
                                    input_template = evnt_range[id2]['input_template'] + [";"] +  evnt_range[id1]['input_template']
                                    output_template = evnt_range[id2]['output_template'] + [";"] +  evnt_range[id1]['output_template']
                                else:
                                    context = evnt_range[id1]['context'][:trg1_end+1] + evnt_range[id2]['context'][trg1_end - 1:]
                                    input_template = evnt_range[id1]['input_template'] + [";"] +  evnt_range[id2]['input_template']
                                    output_template = evnt_range[id1]['output_template'] + [";"] +  evnt_range[id2]['output_template']

                                if len(input_template)+2+2+len(context) <= MAX_LENGTH:
                                    max_tokens = max(len(context) + len(input_template) +2, max_tokens)
                                    max_tgt = max(len(output_template) +1 , max_tgt)
                                    assert(len(output_template) < MAX_TGT_LENGTH)
                                    # tmp = input_template + [" <s>"]+context
                                    # tmp = "".join(tmp)
                                    # print(tmp.replace("Ġ"," "))
                                    # tmp = "".join(output_template)
                                    # print(tmp.replace("Ġ"," "))
                                    # assert 1==0
                                    input_tokens = self.tokenizer.encode_plus(input_template, context, 
                                            add_special_tokens=True,
                                            add_prefix_space=True,
                                            max_length=MAX_LENGTH,
                                            truncation='only_second',
                                            padding='max_length')
                                    tgt_tokens = self.tokenizer.encode_plus(output_template, 
                                    add_special_tokens=True,
                                    add_prefix_space=True, 
                                    max_length=MAX_TGT_LENGTH,
                                    truncation=True,
                                    padding='max_length')

                                    processed_ex = {
                                        'event_idx': i, 
                                        'doc_key': ex['doc_id'], 
                                        'input_token_ids':input_tokens['input_ids'],
                                        'input_attn_mask': input_tokens['attention_mask'],
                                        'tgt_token_ids': tgt_tokens['input_ids'],
                                        'tgt_attn_mask': tgt_tokens['attention_mask'],
                                    }
                                    if split == "test":
                                        writer.write(json.dumps(processed_ex) + '\n')
                                        two_cnt += 1
                                    else:
                                        writer.write(json.dumps(processed_ex) + '\n')
                                        two_cnt += 1
                                        writer.write(json.dumps(processed_ex) + '\n')
                                        two_cnt += 1
                                        writer.write(json.dumps(processed_ex) + '\n')
                                        two_cnt += 1
                                        writer.write(json.dumps(processed_ex) + '\n')
                                        two_cnt += 1
                                        writer.write(json.dumps(processed_ex) + '\n')
                                        two_cnt += 1
                                else:
                                    cnt += 1
                       

            

            print('longest context:{}'.format(max_tokens))
            print('longest target {}'.format(max_tgt))
            print(cnt,one_cnt,two_cnt)
    
    def train_dataloader(self):
        dataset = IEDataset('preprocessed_multievent_{}/train.jsonl'.format(self.hparams.dataset))
        
        dataloader = DataLoader(dataset, 
            pin_memory=True, num_workers=2, 
            collate_fn=my_collate,
            batch_size=self.hparams.train_batch_size, 
            shuffle=True)
        return dataloader 

    
    def val_dataloader(self):
        dataset = IEDataset('preprocessed_multievent_{}/val.jsonl'.format(self.hparams.dataset))
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2, 
            collate_fn=my_collate,
            batch_size=self.hparams.eval_batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        dataset = IEDataset('preprocessed_multievent_{}/test.jsonl'.format(self.hparams.dataset))
        
        dataloader = DataLoader(dataset, pin_memory=True, num_workers=2, 
            collate_fn=my_collate, 
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

    dm = KAIROSDataMultiEventModule(args=args)
    dm.prepare_data() 

    # training dataloader 
    dataloader = dm.train_dataloader() 

    for idx, batch in enumerate(dataloader):
        print(batch)
        break 

    # val dataloader 