import os 
import json 

import torch 
from torch.utils.data import DataLoader, Dataset

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

    return {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,
        'doc_key': doc_keys,
    }

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

class IEDataset(Dataset):
    def __init__(self, input_file):
        super().__init__()
        self.examples = []
        with open(input_file, 'r') as f:
            for line in f:
                ex = json.loads(line.strip())
                self.examples.append(ex)
                # if len(self.examples) == 50:
                #     break
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    

