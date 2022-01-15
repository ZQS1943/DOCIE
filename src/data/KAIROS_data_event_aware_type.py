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

from .data import IEDataset, my_collate_type
from .utils import load_ontology, check_pronoun, clean_mention

MAX_CONTEXT_LENGTH=350 # measured in words
WORDS_PER_EVENT=10 
MAX_LENGTH=512
MAX_TGT_LENGTH=70
TGR_DIS = 20

class KAIROSDataTypeModule(pl.LightningDataModule):
    '''
    Dataset processing for KAIROS. Involves chunking for long documents.
    '''
    def __init__(self, args):
        super().__init__() 
        self.hparams = args
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>',' <tgr>', ' <tag>', ' </tag>'])
        self.role_to_id = {} 
        # convert role to type_id, type_id: 0-UNK, other-role_id
    
    def simple_type(self, type_name):
        _,t1,t2 = type_name.split('.')
        if t2 == "Unspecified":
            return t1.lower()
        if len(t1) < len(t2):
            return t1.lower()
        return t2.lower()

    def get_template(self, ex, index, ontology_dict):
        event_type = ex['event_mentions'][index]['event_type']

        role2arg = defaultdict(list)
        for argument in ex['event_mentions'][index]['arguments']:
            role2arg[argument['role']].append(argument)
        role2arg = dict(role2arg)

        arg_idx2text = defaultdict(list)
        for role in role2arg.keys():
            if role not in ontology_dict[event_type]:
                continue
            for i,argument in enumerate(role2arg[role]):
                arg_text = argument['text']
                if i < len(ontology_dict[event_type][role]):
                    arg_idx = ontology_dict[event_type][role][i]
                else:
                    arg_idx = ontology_dict[event_type][role][-1]
                arg_idx2text[arg_idx].append(arg_text)
        
        template = ontology_dict[event_type]['template']
        input_template = re.sub(r'<arg\d>', '<arg>',template)
        for arg_idx, text_list in arg_idx2text.items():
            text = ' and '.join(text_list)
            template = re.sub('<{}>'.format(arg_idx), text, template)
        output_template = re.sub(r'<arg\d>','<arg>', template)

        space_tokenized_input_template = input_template.split()
        tokenized_input_template = [] 
        for w in space_tokenized_input_template:
            tokenized_input_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))

        space_tokenized_output_template = output_template.split()
        tokenized_output_template = [] 
        for w in space_tokenized_output_template:
            tokenized_output_template.extend(self.tokenizer.tokenize(w, add_prefix_space=True))
        return tokenized_input_template, tokenized_output_template

    def get_context(self, ex, index, max_length):
        '''
        RETURN:  
        context: part of the context with the center word and no more than max length.
        offset: the position of the first token of context in original document
        '''
        trigger = ex['event_mentions'][index]['trigger']
        offset = 0
        context = ex["tokens"]
        center_sent = trigger['sent_idx']
        if len(context) > max_length:
            cur_len = len(ex['sentences'][center_sent][0])
            context = [tup[0] for tup in ex['sentences'][center_sent][0]]
            if cur_len > max_length:
                trigger_start = trigger['start']
                start_idx = max(0, trigger_start - max_length//2)
                end_idx = min(len(context), trigger_start + max_length//2)
                context = context[start_idx: end_idx]
                offset = sum([len(ex['sentences'][idx][0]) for idx in range(center_sent)]) + start_idx
            else:
                left = center_sent -1 
                right = center_sent +1 
                
                total_sents = len(ex['sentences'])
                prev_len = 0 
                while cur_len >  prev_len:
                    prev_len = cur_len 
                    # try expanding the sliding window 
                    if left >= 0:
                        left_sent_tokens = [tup[0] for tup in ex['sentences'][left][0]]
                        if cur_len + len(left_sent_tokens) <= max_length:
                            context = left_sent_tokens + context
                            left -=1 
                            cur_len += len(left_sent_tokens)
                    
                    if right < total_sents:
                        right_sent_tokens = [tup[0] for tup in ex['sentences'][right][0]]
                        if cur_len + len(right_sent_tokens) <= max_length:
                            context = context + right_sent_tokens
                            right +=1 
                            cur_len += len(right_sent_tokens)
                offset = sum([len(ex['sentences'][idx][0]) for idx in range(left+1)])
        
        assert len(context) <= max_length
        assert ex["tokens"][offset:offset + len(context)] == context

        return context, offset

    def tokenize_with_labels(self, tokens, labels, labels2):
        '''
        tokens: a list of tokens
        labels: a list of labels, each of them matches with the token
        RETURN:
        tokenized_tokens: a list of tokenized tokens
        tokenized_labels
        '''
        assert len(tokens) == len(labels) and len(tokens) == len(labels2)
        tokenized_tokens = self.tokenizer.tokenize(' '.join(tokens), add_prefix_space=True)
        tokenized_labels = [0]* len(tokenized_tokens)
        tokenized_labels2 = [0] * len(tokenized_tokens)
        ptr = 0
        for idx,token in enumerate(tokenized_tokens):
            tokenized_labels[idx] = labels[ptr]
            tokenized_labels2[idx] = labels2[ptr]
            if idx+1<len(tokenized_tokens) and (tokenized_tokens[idx+1][0] == "Ġ" or tokenized_tokens[idx+1]==' <tgr>'):
                ptr += 1 
        # for i,_ in enumerate(zip(tokens, labels)):
        #     print(i, _)
        # for i,_ in enumerate(zip(tokenized_tokens, tokenized_labels)):
        #     print(i, _)
        # assert 1==0
        assert len(tokenized_tokens) == len(tokenized_labels) and len(tokenized_tokens) == len(tokenized_labels2)

        return tokenized_tokens, tokenized_labels, tokenized_labels2
        


    def create_instance(self, ex, ontology_dict, index=0, close_events=None, id2entity=None):
        '''
        If close_events is None, we just focus on the conditional generation of ex[index]:
            Input: <s> Template with special <arg> placeholders </s> </s> Passage </s>
            Output: <s> Template with arguments and <arg> when no argument is found
        If close_events is no none, which means there are events we need to be aware of:
            Input: <s> Template with special <arg> placeholders </s> </s> Passage with trigger being highlighted </s>
            Input mask: [1 if it's an argument or trigger of event index else 0 for token in Input]

            Compare:<s> Template with special <arg> placeholders </s> </s> Passage with trigger for event index highlighted and the argument of close events highlighted </s>
            Compare mask: [1 if it's an argument or trigger of event index else 0 for token in Compare]

            Output: <s> Template with arguments and <arg> when no argument is found
        '''
        input_template, output_template = self.get_template(ex, index, ontology_dict)
        context, offset = self.get_context(ex, index, MAX_CONTEXT_LENGTH - WORDS_PER_EVENT * len(close_events))

        trigger = ex['event_mentions'][index]['trigger']
        trigger_start = trigger['start'] - offset
        trigger_end = trigger['end'] - offset
        original_mask = [0]*len(context)
        token_type = [0]*len(context)
        if len(close_events):            

            # mask arguments for E1
            for arg in ex['event_mentions'][index]['arguments']:
                arg_start = id2entity[arg['entity_id']]['start'] - offset
                arg_end = id2entity[arg['entity_id']]['end'] - offset
                if arg_start < 0 or arg_end >= len(context):
                    continue
                for i in range(arg_start, arg_end):
                    original_mask[i] = 1

            # get the tokens need to be tagged (E2)
            for eid in close_events:
                arguments = ex["event_mentions"][eid]["arguments"]
                trigger = ex["event_mentions"][eid]["trigger"]
                for arg in arguments:
                    arg_start = id2entity[arg['entity_id']]['start'] - offset
                    arg_end = id2entity[arg['entity_id']]['end'] - offset
                    if arg_start < 0 or arg_end >= len(context):
                        continue
                    if arg['role'] not in self.role_to_id:
                        self.role_to_id[arg['role']] = len(self.role_to_id) + 1
                    token_type_id = self.role_to_id[arg['role']]
                    for i in range(arg_start, arg_end):
                        token_type[i] = token_type_id


        # get context with trigger being tagged and its argument mask list
        prefix, prefix_labels, prefix_token_type = self.tokenize_with_labels(context[:trigger_start], original_mask[:trigger_start], token_type[:trigger_start])
        tgt, tgt_labels, tgt_token_type = self.tokenize_with_labels(context[trigger_start: trigger_end], original_mask[trigger_start:trigger_end], token_type[trigger_start:trigger_end])
        suffix, suffix_labels, suffix_token_type = self.tokenize_with_labels(context[trigger_end:], original_mask[trigger_end:], token_type[trigger_end:])
        
        context_tag_trigger = prefix + [' <tgr>', ] + tgt + [' <tgr>', ] + suffix
        token_type = prefix_token_type + [0] + tgt_token_type + [0] + suffix_token_type
        if len(close_events):
            context_tag_trigger_mask = prefix_labels + [0] + tgt_labels + [0] + suffix_labels
            return input_template, output_template, context_tag_trigger, token_type, context_tag_trigger_mask
        
        return input_template, output_template, context_tag_trigger, token_type,[]

            
    def prepare_data(self):
        data_dir = self.hparams.data_file
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            ontology_dict = load_ontology(self.hparams.dataset) 
            max_tokens = 0
            max_tgt =0 
            

            for split,f in [('test',self.hparams.test_file), ('train',self.hparams.train_file), ('val',self.hparams.val_file)]:
                cnt = 0
                total_cnt = 0
                print(split)

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
                        
                        event_range = {}
                        for i in range(len(ex['event_mentions'])):
                            start = ex['event_mentions'][i]["trigger"]['start']
                            end =ex['event_mentions'][i]["trigger"]['end']
                            event_range[i] = {'start':start,'end':end}
                        events = event_range.keys()
                        # events = sorted(events, key = lambda x:event_range[x]['start'])


                        for i in range(len(ex['event_mentions'])):
                            if split=='train' and len(ex['event_mentions'][i]['arguments']) ==0:
                                # skip mentions with no arguments 
                                continue 
                            evt_type = ex['event_mentions'][i]['event_type']
                            if evt_type not in ontology_dict: # should be a rare event type 
                                continue 

                            close_events = list(filter(lambda x:abs(event_range[x]['start'] - event_range[i]['start']) <= TGR_DIS and x!=i, events)) # events whose triggers are close to the current trigger
                            # print(i, close_events)
                            
                            input_template, output_template, context_tag_trigger, token_type, context_tag_trigger_mask = self.create_instance(ex, ontology_dict, index=i, close_events=close_events, id2entity=id2entity)
                            
                            max_tokens = max(len(context_tag_trigger) + len(input_template) + 4, max_tokens)
                            max_tgt = max(len(output_template) +1 , max_tgt)
                            assert max_tokens <= MAX_LENGTH
                            assert max_tgt <= MAX_TGT_LENGTH
                            

                            input_tokens = self.tokenizer.encode_plus(input_template,context_tag_trigger, 
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

                                'compare': False,
                                'token_type': [0]*MAX_LENGTH,
                                'compare_mask':[0]*MAX_LENGTH
                            }

                            if len(close_events):                                
                                token_type = [0]*(1 + len(input_template) + 2) + token_type + [0]*(MAX_LENGTH - 3 - len(token_type) - len(input_template))
                                compare_mask = [0]*(1 + len(input_template) + 2) + context_tag_trigger_mask + [0]*(MAX_LENGTH - 3 - len(context_tag_trigger_mask) - len(input_template))

                                processed_ex['token_type'] = token_type
                                processed_ex['compare_mask'] = compare_mask
                                processed_ex['compare'] = True
                                cnt += 1

                                # if split != 'test':
                                #     for _ in range(3):
                                #         writer.write(json.dumps(processed_ex) + '\n')
                                #         total_cnt += 1
                                #         cnt += 1

                            # tokens = self.tokenizer.convert_ids_to_tokens(processed_ex["input_token_ids"])
                            # input_1 = "".join(tokens)
                            # print("input_1:", input_1.replace("Ġ"," "))
                            # for i,_ in enumerate(zip(tokens, processed_ex['token_type'], processed_ex['compare_mask'])):
                            #     if _[1] == 0 and _[2] == 0:
                            #         print(f'{i}\t{_[0]}')
                            #     else:
                            #         print(f'{i}\t{_[0]}\t{_[1]}\t{_[2]}')
                            # if cnt == 5:
                            #     assert 1==0
                            # print("_"*50)
                            writer.write(json.dumps(processed_ex) + '\n')
                            total_cnt += 1
                            
                print(f'total_event: {total_cnt}')
                print(f'special event: {cnt}')
                print(self.role_to_id)

            print('longest context:{}'.format(max_tokens))
            print('longest target {}'.format(max_tgt))
    
    def train_dataloader(self):
        dataset = IEDataset(f'{self.hparams.data_file}/train.jsonl')
        
        dataloader = DataLoader(dataset, 
            pin_memory=False, num_workers=2, 
            collate_fn=my_collate_type,
            batch_size=self.hparams.train_batch_size, 
            shuffle=True)
        return dataloader 

    
    def val_dataloader(self):
        dataset = IEDataset(f'{self.hparams.data_file}/val.jsonl')
        
        dataloader = DataLoader(dataset, pin_memory=False, num_workers=2, 
            collate_fn=my_collate_type,
            batch_size=self.hparams.eval_batch_size, shuffle=False)
        return dataloader

    def test_dataloader(self):
        dataset = IEDataset(f'{self.hparams.data_file}/test.jsonl')
        
        dataloader = DataLoader(dataset, pin_memory=False, num_workers=2, 
            collate_fn=my_collate_type, 
            batch_size=self.hparams.eval_batch_size, shuffle=False)

        return dataloader


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
    parser.add_argument('--data_file', default='./preprocessed_type',type=str)
    args = parser.parse_args() 

    dm = KAIROSDataEventAwareModule(args=args)
    dm.prepare_data() 

    # training dataloader 
    dataloader = dm.train_dataloader() 

    for idx, batch in enumerate(dataloader):
        print(batch)
        break 

    # val dataloader 