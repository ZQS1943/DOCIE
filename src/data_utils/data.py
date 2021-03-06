import os 
import json 

import torch 
from torch.utils.data import DataLoader, Dataset
import random

def my_collate_seq(batch):

    doc_keys = [ex['doc_key'] for ex in batch]
    input_token_ids = torch.stack([torch.LongTensor(ex['input_token_ids']) for ex in batch]) 
    input_attn_mask = torch.stack([torch.BoolTensor(ex['input_attn_mask']) for ex in batch])
    labels = torch.stack([torch.LongTensor(ex['labels']) for ex in batch]) 
    
    return {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'doc_key': doc_keys,
        'labels': labels
    }

def my_collate_type(batch):
    # processed_ex = {
    #                             'event_idx': i, 
    #                             'doc_key': ex['doc_id'], 
    #                             'input_token_ids':input_tokens['input_ids'],
    #                             'input_attn_mask': input_tokens['attention_mask'],
    #                             'tgt_token_ids': tgt_tokens['input_ids'],
    #                             'tgt_attn_mask': tgt_tokens['attention_mask'],

    #                             'compare': False,
    #                             'token_type': [0]*MAX_LENGTH,
    #                             'compare_mask':[0]*MAX_LENGTH
    #                         }
    doc_keys = [ex['doc_key'] for ex in batch]
    input_token_ids = torch.stack([torch.LongTensor(ex['input_token_ids']) for ex in batch]) 
    input_attn_mask = torch.stack([torch.BoolTensor(ex['input_attn_mask']) for ex in batch])
    tgt_token_ids = torch.stack([torch.LongTensor(ex['tgt_token_ids']) for ex in batch]) 
    tgt_attn_mask = torch.stack([torch.BoolTensor(ex['tgt_attn_mask']) for ex in batch])

    compare = []
    token_type = torch.zeros_like(input_token_ids)
    compare_mask  = []
    if 'compare' in batch[0]:
        # print('haha')
        compare = [ex['compare'] for ex in batch]
        # token_type = torch.stack([torch.LongTensor(ex['token_type']) for ex in batch])
        compare_mask = torch.stack([torch.BoolTensor(ex['compare_mask']) for ex in batch])


    return {
        'doc_key': doc_keys,
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,

        'compare_mask': compare_mask,
        # 'token_type': token_type,
        'token_type': None,
        'compare': compare,
    }

def my_collate(batch):
    '''
    'doc_key': ex['doc_key'],
    'input_token_ids':input_tokens['input_ids'],
    'input_attn_mask': input_tokens['attention_mask'],
    'tgt_token_ids': tgt_tokens['input_ids'],
    'tgt_attn_mask': tgt_tokens['attention_mask'],
    '''
    
    
    doc_keys = [ex['doc_key'] for ex in batch]
    input_token_ids = torch.stack([torch.LongTensor(ex['input_token_ids']) for ex in batch]) 
    input_attn_mask = torch.stack([torch.BoolTensor(ex['input_attn_mask']) for ex in batch])
    tgt_token_ids = torch.stack([torch.LongTensor(ex['tgt_token_ids']) for ex in batch]) 
    tgt_attn_mask = torch.stack([torch.BoolTensor(ex['tgt_attn_mask']) for ex in batch])

    result = {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,
        'doc_key': doc_keys,
    }
    if 'event_idx' in batch[0]:
        event_idx = [ex['event_idx'] for ex in batch]
        result['event_idx'] = event_idx
    if 'event_type' in batch[0]:
        event_type = [ex['event_type'] for ex in batch]
        result['event_type'] = event_type
    if 'trigger' in batch[0]:
        trigger = [ex['trigger'] for ex in batch]
        result['trigger'] = trigger

    return result

def my_collate_finetune(batch):
    '''
    processed_ex = {
                            'event_idx',
                            'doc_key': ex['doc_key'],
                            'input_token_ids_1',
                            'input_attn_mask_1',
                            'argument_mask_1',
                            'input_token_ids_2',
                            'input_attn_mask_2',
                            'argument_mask_2',
                        }
    '''
    doc_keys = [ex['doc_key'] for ex in batch]
    input_token_ids_1 = torch.stack([torch.LongTensor(ex['input_token_ids_1']) for ex in batch]) 
    input_attn_mask_1 = torch.stack([torch.BoolTensor(ex['input_attn_mask_1']) for ex in batch])
    argument_mask_1 = torch.stack([torch.BoolTensor(ex['argument_mask_1']) for ex in batch])
    input_token_ids_2 = torch.stack([torch.LongTensor(ex['input_token_ids_2']) for ex in batch]) 
    input_attn_mask_2 = torch.stack([torch.BoolTensor(ex['input_attn_mask_2']) for ex in batch])
    argument_mask_2 = torch.stack([torch.BoolTensor(ex['argument_mask_2']) for ex in batch])

    return {
        'input_token_ids_1': input_token_ids_1,
        'input_attn_mask_1': input_attn_mask_1,
        'argument_mask_1': argument_mask_1,
        'input_token_ids_2': input_token_ids_2,
        'input_attn_mask_2': input_attn_mask_2,
        'argument_mask_2': argument_mask_2,
        'doc_key': doc_keys,
    }

def my_collate_multitask(batch):
    '''
    'event_idx',
    'input_token_ids':input_tokens['input_ids'],
    'input_attn_mask': input_tokens['attention_mask'],
    'tgt_token_ids': tgt_tokens['input_ids'],
    'tgt_attn_mask': tgt_tokens['attention_mask'],
                            'doc_key': ex['doc_key'],
                            'input_token_ids_1',
                            'input_attn_mask_1',
                            'argument_mask_1',
                            'input_token_ids_2',
                            'input_attn_mask_2',
                            'argument_mask_2',
    ''
    '''
    doc_keys = [ex['doc_key'] for ex in batch]
    input_token_ids = torch.stack([torch.LongTensor(ex['input_token_ids']) for ex in batch]) 
    input_attn_mask = torch.stack([torch.BoolTensor(ex['input_attn_mask']) for ex in batch])
    tgt_token_ids = torch.stack([torch.LongTensor(ex['tgt_token_ids']) for ex in batch]) 
    tgt_attn_mask = torch.stack([torch.BoolTensor(ex['tgt_attn_mask']) for ex in batch])

    input_token_ids_1 = torch.stack([torch.LongTensor(ex['input_token_ids_1']) for ex in batch]) 
    input_attn_mask_1 = torch.stack([torch.BoolTensor(ex['input_attn_mask_1']) for ex in batch])
    argument_mask_1 = torch.stack([torch.BoolTensor(ex['argument_mask_1']) for ex in batch])
    input_token_ids_2 = torch.stack([torch.LongTensor(ex['input_token_ids_2']) for ex in batch]) 
    input_attn_mask_2 = torch.stack([torch.BoolTensor(ex['input_attn_mask_2']) for ex in batch])
    argument_mask_2 = torch.stack([torch.BoolTensor(ex['argument_mask_2']) for ex in batch])

    return {
        'input_token_ids_1': input_token_ids_1,
        'input_attn_mask_1': input_attn_mask_1,
        'argument_mask_1': argument_mask_1,
        'input_token_ids_2': input_token_ids_2,
        'input_attn_mask_2': input_attn_mask_2,
        'argument_mask_2': argument_mask_2,
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,
        'doc_key': doc_keys,
    }

def my_collate_event_aware(batch):
    '''
    processed_ex = {
                                'event_idx': i, 
                                'doc_key': ex['doc_id'], 
                                'input_token_ids':input_tokens['input_ids'],
                                'input_attn_mask': input_tokens['attention_mask'],
                                'tgt_token_ids': tgt_tokens['input_ids'],
                                'tgt_attn_mask': tgt_tokens['attention_mask'],

                                'compare': False,
                                'input_mask': [0]*MAX_LENGTH,
                                'compare_token_ids': [0]*MAX_LENGTH,
                                'compare_attn_mask': [0]*MAX_LENGTH,
                                'compare_mask': [0]*MAX_LENGTH
                            }
    '''
    doc_keys = [ex['doc_key'] for ex in batch]
    input_token_ids = torch.stack([torch.LongTensor(ex['input_token_ids']) for ex in batch]) 
    input_attn_mask = torch.stack([torch.BoolTensor(ex['input_attn_mask']) for ex in batch])
    tgt_token_ids = torch.stack([torch.LongTensor(ex['tgt_token_ids']) for ex in batch]) 
    tgt_attn_mask = torch.stack([torch.BoolTensor(ex['tgt_attn_mask']) for ex in batch])

    compare = [ex['compare'] for ex in batch]
    input_mask = torch.stack([torch.BoolTensor(ex['input_mask']) for ex in batch])
    input_mask = input_mask[compare]
    compare_token_ids = torch.stack([torch.LongTensor(ex['compare_token_ids']) for ex in batch]) 
    compare_token_ids = compare_token_ids[compare]
    compare_attn_mask = torch.stack([torch.BoolTensor(ex['compare_attn_mask']) for ex in batch])
    compare_attn_mask = compare_attn_mask[compare]
    compare_mask = torch.stack([torch.BoolTensor(ex['compare_mask']) for ex in batch])
    compare_mask = compare_mask[compare]
   
    return {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'input_mask': input_mask,
        'compare_token_ids': compare_token_ids,
        'compare_attn_mask': compare_attn_mask,
        'compare_mask': compare_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,
        'doc_key': doc_keys,
        'compare': compare,
    }

def my_collate_comparing(batch):
    # 'event_idx': i, 
    # 'doc_key': ex['doc_id'], 
    # 'input_token_ids':input_tokens['input_ids'],
    # 'input_attn_mask': input_tokens['attention_mask'],
    # 'tgt_token_ids': tgt_tokens['input_ids'],
    # 'tgt_attn_mask': tgt_tokens['attention_mask'],
    # 'compare_token_ids': compare_tokens['input_ids'],
    # 'compare_attn_mask': compare_tokens['attention_mask'],
    # 'input_mask' : input_mask,
    # 'compare_mask':compare_mask

    doc_keys = [ex['doc_key'] for ex in batch]
    input_token_ids = torch.stack([torch.LongTensor(ex['input_token_ids']) for ex in batch]) 
    input_attn_mask = torch.stack([torch.BoolTensor(ex['input_attn_mask']) for ex in batch])
    tgt_token_ids = torch.stack([torch.LongTensor(ex['tgt_token_ids']) for ex in batch]) 
    tgt_attn_mask = torch.stack([torch.BoolTensor(ex['tgt_attn_mask']) for ex in batch])

    compare_token_ids = torch.stack([torch.LongTensor(ex['compare_token_ids']) for ex in batch])
    compare_attn_mask = torch.stack([torch.BoolTensor(ex['compare_attn_mask']) for ex in batch])

    input_mask = torch.stack([torch.BoolTensor(ex['input_mask']) for ex in batch])
    compare_mask = torch.stack([torch.BoolTensor(ex['compare_mask']) for ex in batch])

    return {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,
        'compare_token_ids': compare_token_ids,
        'compare_attn_mask': compare_attn_mask,
        'input_mask' : input_mask,
        'compare_mask':compare_mask,
        'doc_key': doc_keys,
    }



def my_collate_event_aware_only_arg(batch):
    '''
    processed_ex = {
                                'event_idx': i, 
                                'doc_key': ex['doc_id'], 
                                'input_token_ids':input_tokens['input_ids'],
                                'input_attn_mask': input_tokens['attention_mask'],
                                'tgt_token_ids': tgt_tokens['input_ids'],
                                'tgt_attn_mask': tgt_tokens['attention_mask'],

                                'compare': False,
                                'input_mask': [0]*MAX_LENGTH,
                                'compare_token_ids': [0]*MAX_LENGTH,
                                'compare_attn_mask': [0]*MAX_LENGTH,
                                'compare_mask': [0]*MAX_LENGTH
                            }
    '''
    # print('my_collate_event_aware_only_arg')
    # print(batch[0].keys())
    doc_keys = [ex['doc_key'] for ex in batch]
    # input_token_ids = torch.stack([torch.LongTensor(ex['input_token_ids']) for ex in batch]) 
    # input_attn_mask = torch.stack([torch.BoolTensor(ex['input_attn_mask']) for ex in batch])
    tgt_token_ids = torch.stack([torch.LongTensor(ex['tgt_token_ids']) for ex in batch]) 
    tgt_attn_mask = torch.stack([torch.BoolTensor(ex['tgt_attn_mask']) for ex in batch])

    # compare = [ex['compare'] for ex in batch]
    # input_mask = torch.stack([torch.BoolTensor(ex['input_mask']) for ex in batch])
    # input_mask = input_mask[compare]
    # compare_token_ids = torch.stack([torch.LongTensor(ex['compare_token_ids']) for ex in batch]) 
    # compare_token_ids = compare_token_ids[compare]
    # compare_attn_mask = torch.stack([torch.BoolTensor(ex['compare_attn_mask']) for ex in batch])
    # compare_attn_mask = compare_attn_mask[compare]
    # compare_mask = torch.stack([torch.BoolTensor(ex['compare_mask']) for ex in batch])
    # compare_mask = compare_mask[compare]
   
    input_token_ids = torch.stack([torch.LongTensor(ex['compare_token_ids']) if ex['compare'] else torch.LongTensor(ex['input_token_ids']) for ex in batch])
    input_attn_mask = torch.stack([torch.BoolTensor(ex['compare_attn_mask']) if ex['compare'] else torch.BoolTensor(ex['input_attn_mask']) for ex in batch])
    return {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,
        'doc_key': doc_keys,
    }

class IEDataset(Dataset):
    def __init__(self, input_file, tokenizer = None):
        super().__init__()
        self.examples = []
        with open(input_file, 'r') as f:
            for line in f:
                ex = json.loads(line.strip())
                self.examples.append(ex)
                # if len(self.examples) == 50:
                #     break
        # if toksenizer != None:
        #     print('*'*10)
        #     print(f'INIT DATA FILE: {input_file}')
        #     selected_ex = random.randint(0,len(self.examples))
        #     print(f'sample:{selected_ex}')
        #     print('input:')
        #     print(tokenizer.decode(self.examples[selected_ex]['input_token_ids'], skip_special_tokens=True),)
        #     print('output:')
        #     print(tokenizer.decode(self.examples[selected_ex]['tgt_token_ids'], skip_special_tokens=True),)
        #     print('*'*10)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    

