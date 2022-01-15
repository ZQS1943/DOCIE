import os 
import json 
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
from .data_utils import create_instance_tag_other_event, create_instance_tag_comparing, create_instance_normal,create_instance_normal_sentence_selection,create_instance_tag_other_event_sentence_selection, create_instance_two_event,create_instance_seq

from tqdm import tqdm

MAX_CONTEXT_LENGTH=350 # measured in words
WORDS_PER_EVENT=10 
MAX_LENGTH=512
MAX_TGT_LENGTH=70

def get_data_normal_sentence_selection(source = None, target = None, tokenizer = None, coref = None):
    ontology_dict = load_ontology("KAIROS") 
    max_tokens = 0
    max_tgt =0 
    print(f"source:{source}")
    print(f"target:{target}")

    with open(source, 'r') as reader, open(coref, 'r') as coref_reader , open(target, 'w') as writer:
        total_cnt = 0
        arg_tgr_not_same_sent_cnt = 0


        for line, coref_line in tqdm(zip(reader,coref_reader)):
            ex = json.loads(line.strip())
            ex_coref = json.loads(coref_line.strip())
            assert ex['doc_id'] == ex_coref['doc_key']

            entity2coref = {}
            for coref, entity_list in enumerate(ex_coref['clusters']):
                for entity in entity_list:
                    entity2coref[entity] = coref

            sent2entity = defaultdict(set)
            entity2sent = defaultdict(set)
            id2entity = {}
            for entity in ex["entity_mentions"]:
                id2entity[entity["id"]] = entity
                sent2entity[entity['sent_idx']].add(entity2coref[entity['id']] if entity['id'] in entity2coref else entity['id']) 
                entity2sent[entity2coref[entity['id']] if entity['id'] in entity2coref else entity['id']].add(entity['sent_idx'])
            
            related_sent = defaultdict(set)
            for sent in range(len(ex['sentences'])):
                related_sent[sent].add(sent)
                if sent - 1 >= 0:
                    related_sent[sent].add(sent - 1)
                if sent + 1 < len(ex['sentences']):
                    related_sent[sent].add(sent + 1)
                for entity in list(sent2entity[sent]):
                    related_sent[sent].update(entity2sent[entity])
            # print(related_sent)

            for i in range(len(ex['event_mentions'])):
                trigger = ex['event_mentions'][i]['trigger']
                center_sent = trigger['sent_idx']
                for arg in ex['event_mentions'][i]['arguments']:
                    if abs(id2entity[arg['entity_id']]['sent_idx'] - center_sent) > 1:
                        arg_tgr_not_same_sent_cnt += 1
                        # print(center_sent, id2entity[arg['entity_id']]['sent_idx'])
                        break
                        
                    
                evt_type = ex['event_mentions'][i]['event_type']
                
                assert evt_type in ontology_dict 

                input_template, output_template, context_tag_trigger = create_instance_normal_sentence_selection(ex, ontology_dict, index=i, tokenizer = tokenizer, id2entity=id2entity, related_sent = related_sent)
                
                max_tokens = max(len(context_tag_trigger) + len(input_template) + 4, max_tokens)
                max_tgt = max(len(output_template) +1 , max_tgt)
                # print(len(context_tag_other_events), len(input_template), len(context_tag_trigger))
                # print(max_tokens, len(context_tag_trigger), len(input_template))
                # assert max_tokens <= MAX_LENGTH
                assert max_tgt <= MAX_TGT_LENGTH
                

                input_tokens = tokenizer.encode_plus(input_template,context_tag_trigger, 
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=MAX_LENGTH,
                truncation='only_second',
                padding='max_length')

                tgt_tokens = tokenizer.encode_plus(output_template, 
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
                    'tgt_attn_mask': tgt_tokens['attention_mask']
                }

                # tokens = tokenizer.decode(processed_ex["input_token_ids"], skip_special_tokens=True)
                
                writer.write(json.dumps(processed_ex) + '\n')
                total_cnt += 1
    print(f'total_event: {total_cnt}')
    print(f'arg_tgr_not_same_sent_event: {arg_tgr_not_same_sent_cnt}')

    print('longest context:{}'.format(max_tokens))
    print('longest target {}'.format(max_tgt))    


def get_data_tag_only_sentence_selection(source = None, target = None, tokenizer = None, coref = None):
    ontology_dict = load_ontology("KAIROS") 
    max_tokens = 0
    max_tgt =0 
    print(f"source:{source}")
    print(f"target:{target}")

    with open(source, 'r') as reader, open(coref, 'r') as coref_reader , open(target, 'w') as writer:
        total_cnt = 0
        arg_tgr_not_same_sent_cnt = 0


        for line, coref_line in tqdm(zip(reader,coref_reader)):
            ex = json.loads(line.strip())
            ex_coref = json.loads(coref_line.strip())
            assert ex['doc_id'] == ex_coref['doc_key']

            entity2coref = {}
            for coref, entity_list in enumerate(ex_coref['clusters']):
                for entity in entity_list:
                    entity2coref[entity] = coref

            sent2entity = defaultdict(set)
            entity2sent = defaultdict(set)
            
            id2entity = {}
            for entity in ex["entity_mentions"]:
                id2entity[entity["id"]] = entity
                sent2entity[entity['sent_idx']].add(entity2coref[entity['id']] if entity['id'] in entity2coref else entity['id']) 
                entity2sent[entity2coref[entity['id']] if entity['id'] in entity2coref else entity['id']].add(entity['sent_idx'])
           
            related_sent = defaultdict(set)

            for sent in range(len(ex['sentences'])):
                related_sent[sent].add(sent)
                if sent - 1 >= 0:
                    related_sent[sent].add(sent - 1)
                if sent + 1 < len(ex['sentences']):
                    related_sent[sent].add(sent + 1)
                for entity in list(sent2entity[sent]):
                    related_sent[sent].update(entity2sent[entity])


            sent2offset = {}
            offest = 0
            for idx, sent in enumerate(ex['sentences']):
                sent2offset[idx] = offest
                offest += len(sent[0])
            
            def get_sent_idx(start):
                sent_idx = 0
                while sent_idx + 1 < len(sent2offset):
                    if sent2offset[sent_idx + 1] > start:
                        return sent_idx
                    sent_idx += 1                
                return sent_idx

            sent2args = defaultdict(list)
            for idx,event in enumerate(ex['event_mentions']):
                for arg in event['arguments']:
                    if 'entity_id' in arg:
                        arg_start = id2entity[arg['entity_id']]['start']
                        arg_end = id2entity[arg['entity_id']]['end']
                    else:
                        arg_start = arg['start']
                        arg_end = arg['end']
                    sent_idx = get_sent_idx(arg_start)                    
                    sent2args[sent_idx].append({
                        'start': arg_start,
                        'end': arg_end,
                        'eid': idx,
                        'role': arg['role']
                    })
                    

            for i in range(len(ex['event_mentions'])):
                trigger = ex['event_mentions'][i]['trigger']
                center_sent = trigger['sent_idx']
                for arg in ex['event_mentions'][i]['arguments']:
                    if 'entity_id' in arg:
                        arg_start = id2entity[arg['entity_id']]['start']
                        arg_end = id2entity[arg['entity_id']]['end']
                    else:
                        arg_start = arg['start']
                        arg_end = arg['end']
                    sent_idx = get_sent_idx(arg_start) 
                    if abs(sent_idx - center_sent) > 1:
                        arg_tgr_not_same_sent_cnt += 1
                        # print(center_sent, id2entity[arg['entity_id']]['sent_idx'])
                        break
                    
                evt_type = ex['event_mentions'][i]['event_type']
                
                assert evt_type in ontology_dict 

                input_template, output_template, context_tag_trigger = create_instance_tag_other_event_sentence_selection(ex, ontology_dict, index=i, tokenizer = tokenizer, id2entity=id2entity, related_sent = related_sent, sent2args = sent2args)
                
                max_tokens = max(len(context_tag_trigger) + len(input_template) + 4, max_tokens)
                max_tgt = max(len(output_template) +1 , max_tgt)
                # print(len(context_tag_other_events), len(input_template), len(context_tag_trigger))
                # print(max_tokens, len(context_tag_trigger), len(input_template))
                # assert max_tokens <= MAX_LENGTH
                assert max_tgt <= MAX_TGT_LENGTH
                

                input_tokens = tokenizer.encode_plus(input_template,context_tag_trigger, 
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=MAX_LENGTH,
                truncation='only_second',
                padding='max_length')

                tgt_tokens = tokenizer.encode_plus(output_template, 
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
                    'tgt_attn_mask': tgt_tokens['attention_mask']
                }

                # tokens = tokenizer.decode(processed_ex["input_token_ids"], skip_special_tokens=True)
                
                writer.write(json.dumps(processed_ex) + '\n')
                total_cnt += 1
    print(f'total_event: {total_cnt}')
    print(f'arg_tgr_not_same_sent_event: {arg_tgr_not_same_sent_cnt}')

    print('longest context:{}'.format(max_tokens))
    print('longest target {}'.format(max_tgt))    

def get_data_seq(source = None, target = None, tokenizer = None, dataset = "KAIROS"):
    ontology_dict = load_ontology(dataset) 
    # print(ontology_dict)
    roles_list = []
    for et in ontology_dict:
        roles_list.extend(ontology_dict[et]['roles'])
    roles_list = sorted(list(set(roles_list)))
    roles_to_label = {role:i + 1 for i,role in enumerate(roles_list)}
    label_to_role = {i + 1:role for i,role in enumerate(roles_list)}

    max_tokens = 0
    max_tgt =0 
    print(f"source:{source}")
    print(f"target:{target}")
    print(f"dataset:{dataset}")
    print(f"role_numbers:{len(roles_list)}")

    with open(source, 'r') as reader , open(target, 'w') as writer:
        total_cnt = 0
        for line in tqdm(reader):
            ex = json.loads(line.strip())

            if 'entity_mentions' in ex:
                id2entity = {}
                for entity in ex["entity_mentions"]:
                    id2entity[entity["id"]] = entity
            else:
                id2entity = None

            for i in range(len(ex['event_mentions'])):
                evt_type = ex['event_mentions'][i]['event_type']
                
                assert evt_type in ontology_dict 
                # if len(ex['event_mentions'][i]['arguments']) == 0:
                #     continue

                context_tag_trigger, labels = create_instance_seq(ex, ontology_dict, index=i, tokenizer = tokenizer, id2entity=id2entity, roles_to_label = roles_to_label)
                
                max_tokens = max(len(context_tag_trigger), max_tokens)

                input_tokens = tokenizer.encode_plus(context_tag_trigger, 
                add_special_tokens=True,
                add_prefix_space=True, 
                max_length=MAX_LENGTH,
                truncation=True,
                padding='max_length')

                labels = [0] + labels
                if len(labels) > MAX_LENGTH:
                    labels = labels[:MAX_LENGTH]
            
                processed_ex = {
                    'event_idx': i, 
                    'doc_key': ex['doc_id'], 
                    'input_token_ids':input_tokens['input_ids'],
                    'input_attn_mask': input_tokens['attention_mask'],
                    'labels': labels + [0] * (MAX_LENGTH - len(labels))
                }
                
                writer.write(json.dumps(processed_ex) + '\n')
                total_cnt += 1
    print(f'total_event: {total_cnt}')

    print('longest context:{}'.format(max_tokens))
    print('longest target {}'.format(max_tgt)) 
    return label_to_role


def get_data_normal(source = None, target = None, tokenizer = None, dataset = "KAIROS"):
    ontology_dict = load_ontology(dataset) 
    max_tokens = 0
    max_tgt =0 
    print(f"source:{source}")
    print(f"target:{target}")
    print(f"dataset:{dataset}")

    with open(source, 'r') as reader , open(target, 'w') as writer:
        total_cnt = 0


        for line in tqdm(reader):
            ex = json.loads(line.strip())

            if 'entity_mentions' in ex:
                id2entity = {}
                for entity in ex["entity_mentions"]:
                    id2entity[entity["id"]] = entity
            else:
                id2entity = None

            for i in range(len(ex['event_mentions'])):
                evt_type = ex['event_mentions'][i]['event_type']
                
                assert evt_type in ontology_dict 

                input_template, output_template, context_tag_trigger = create_instance_normal(ex, ontology_dict, index=i, tokenizer = tokenizer, id2entity=id2entity)
                
                max_tokens = max(len(context_tag_trigger) + len(input_template) + 4, max_tokens)
                max_tgt = max(len(output_template) +1 , max_tgt)
                # print(len(context_tag_other_events), len(input_template), len(context_tag_trigger))
                # print(max_tokens, len(context_tag_trigger), len(input_template))
                # assert max_tokens <= MAX_LENGTH
                assert max_tgt <= MAX_TGT_LENGTH
                

                input_tokens = tokenizer.encode_plus(input_template,context_tag_trigger, 
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=MAX_LENGTH,
                truncation='only_second',
                padding='max_length')

                tgt_tokens = tokenizer.encode_plus(output_template, 
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
                    'tgt_attn_mask': tgt_tokens['attention_mask']
                }

                # tokens = tokenizer.decode(processed_ex["input_token_ids"], skip_special_tokens=True)
                
                writer.write(json.dumps(processed_ex) + '\n')
                total_cnt += 1
    print(f'total_event: {total_cnt}')

    print('longest context:{}'.format(max_tokens))
    print('longest target {}'.format(max_tgt))    

def get_data_tag_only(source = None, target = None, tokenizer = None, trigger_dis = 40, dataset='KAIROS'):            
    ontology_dict = load_ontology(dataset) 
    max_tokens = 0
    max_tgt =0 
    print(f"source:{source}")
    print(f"target:{target}")
    print(f'trigger_dis:{trigger_dis}')
    print(f'dataset:{dataset}')

    with open(source, 'r') as reader , open(target, 'w') as writer:
        total_cnt = 0
        cnt = 0


        for line in tqdm(reader):
            ex = json.loads(line.strip())

            if 'entity_mentions' in ex:
                id2entity = {}
                for entity in ex["entity_mentions"]:
                    id2entity[entity["id"]] = entity
            else:
                id2entity = None
                        
            event_range = {}
            for i in range(len(ex['event_mentions'])):
                # if len(ex['event_mentions'][i]['arguments']) > 0:
                start = ex['event_mentions'][i]["trigger"]['start']
                end =ex['event_mentions'][i]["trigger"]['end']
                event_range[i] = {'start':start,'end':end}
            events = event_range.keys()

            for i in range(len(ex['event_mentions'])):
                evt_type = ex['event_mentions'][i]['event_type']
                
                assert evt_type in ontology_dict 

                close_events = list(filter(lambda x:abs(event_range[x]['start'] - event_range[i]['start']) <= trigger_dis and x!=i, events)) # events whose triggers are close to the current trigger
                if len(close_events):
                    cnt += 1
                input_template, output_template, context_tag_trigger = create_instance_tag_other_event(ex, ontology_dict, index=i, close_events=close_events, tokenizer = tokenizer, id2entity=id2entity)

                # if len(input_template) + len(context_tag_trigger) + 4 > MAX_LENGTH:
                #     print(len(input_template), len(context_tag_trigger))
                #     tgr_idx = context_tag_trigger.index(' <tgr>')
                #     truncate_length = len(input_template) + len(context_tag_trigger) + 4 - MAX_LENGTH
                #     print('truncate length: ',truncate_length)
                #     left = True
                #     if len(context_tag_trigger) - tgr_idx > tgr_idx:
                #         left = False
                #     if left:
                #         context_tag_trigger = context_tag_trigger[truncate_length:]
                #     else:
                #         context_tag_trigger = context_tag_trigger[:-truncate_length]
                #     print(len(input_template), len(context_tag_trigger))
                
                max_tokens = max(len(context_tag_trigger) + len(input_template) + 4, max_tokens)
                max_tgt = max(len(output_template) +1 , max_tgt)
                # print(len(context_tag_other_events), len(input_template), len(context_tag_trigger))
                # print(max_tokens, len(context_tag_trigger), len(input_template))
                # assert max_tokens <= MAX_LENGTH
                assert max_tgt <= MAX_TGT_LENGTH
                
                # print(input_template, context_tag_trigger)
                input_tokens = tokenizer.encode_plus(input_template,context_tag_trigger, 
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=MAX_LENGTH,
                truncation='only_second',
                padding='max_length')

                tgt_tokens = tokenizer.encode_plus(output_template, 
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
                    'tgt_attn_mask': tgt_tokens['attention_mask']
                }

                # tokens = tokenizer.decode(processed_ex["input_token_ids"], skip_special_tokens=True)
                
                writer.write(json.dumps(processed_ex) + '\n')
                total_cnt += 1
    print(f'total_event: {total_cnt}')
    print(f'has close events: {cnt}')

    print('longest context:{}'.format(max_tokens))
    print('longest target {}'.format(max_tgt))

def get_data_two_event(source = None, target = None, tokenizer = None):            
    ontology_dict = load_ontology("KAIROS") 
    max_tokens = 0
    max_tgt =0 
    print(f"source:{source}")
    print(f"target:{target}")

    with open(source, 'r') as reader , open(target, 'w') as writer:
        total_cnt = 0
        cnt = 0


        for line in tqdm(reader):
            ex = json.loads(line.strip())

            if 'entity_mentions' in ex:
                id2entity = {}
                for entity in ex["entity_mentions"]:
                    id2entity[entity["id"]] = entity
            else:
                id2entity = None
                        
            event_range = {}
            for i in range(len(ex['event_mentions'])):
                # if len(ex['event_mentions'][i]['arguments']) > 0:
                start = ex['event_mentions'][i]["trigger"]['start']
                end =ex['event_mentions'][i]["trigger"]['end']
                event_range[i] = {'start':start,'end':end}
            events = event_range.keys()
            event_mentions = sorted(zip(ex['event_mentions'], range(len(ex['event_mentions']))), key=lambda x:x[0]['trigger']['start'])

            for i,j in zip(range(0, len(event_mentions) - 1),range(1, len(event_mentions))):
                i = event_mentions[i][1]
                j = event_mentions[j][1]
                evt_type_i = ex['event_mentions'][i]['event_type']
                evt_type_j = ex['event_mentions'][j]['event_type']
                
                assert evt_type_i in ontology_dict and evt_type_j in ontology_dict 
                
                input_template, output_template, context_tag_trigger = create_instance_two_event(ex, ontology_dict, index_i=i, index_j=j, tokenizer = tokenizer, id2entity=id2entity)
                
                max_tokens = max(len(context_tag_trigger) + len(input_template) + 4, max_tokens)
                max_tgt = max(len(output_template) +1 , max_tgt)
                # print(len(context_tag_other_events), len(input_template), len(context_tag_trigger))
                # print(max_tokens, len(context_tag_trigger), len(input_template))
                # assert max_tokens <= MAX_LENGTH
                # assert max_tgt <= MAX_TGT_LENGTH
                
                # print(input_template, context_tag_trigger)
                input_tokens = tokenizer.encode_plus(input_template,context_tag_trigger, 
                add_special_tokens=True,
                add_prefix_space=True,
                max_length=MAX_LENGTH,
                truncation='only_second',
                padding='max_length')



                tgt_tokens = tokenizer.encode_plus(output_template, 
                add_special_tokens=True,
                add_prefix_space=True, 
                max_length=MAX_TGT_LENGTH,
                truncation=True,
                padding='max_length')
            
                processed_ex = {
                    'event_idx': (i,j),
                    'event_type': (evt_type_i, evt_type_j), 
                    'trigger': (event_range[i],event_range[j]),
                    'doc_key': ex['doc_id'], 
                    'input_token_ids':input_tokens['input_ids'],
                    'input_attn_mask': input_tokens['attention_mask'],
                    'tgt_token_ids': tgt_tokens['input_ids'],
                    'tgt_attn_mask': tgt_tokens['attention_mask']
                }

                # tokens = tokenizer.convert_ids_to_tokens(processed_ex["tgt_token_ids"])
                # print(output_template)
                # print(tokens)
                # assert 1==0
                
                writer.write(json.dumps(processed_ex) + '\n')
                total_cnt += 1
    print(f'total_event: {total_cnt}')
    print(f'has close events: {cnt}')

    print('longest context:{}'.format(max_tokens))
    print('longest target {}'.format(max_tgt))
     

def get_data_tag_comparing(source = None, target = None, tokenizer = None, trigger_dis = 40, dataset='KAIROS'):          
    # in this function, we try to convert the input original data file into:
    # For each event:
    #   'event_idx'
    #   'doc_key'
    #   'input_token_ids':template + context with trigger highlighted
    #   'input_attn_mask'
    #   'tgt_token_ids': gold template
    #   'tgt_attn_mask'

    #   'compare_token_ids': template + context with trigger and arguments of other events highlighted
    #   'compare_attn_mask'
    #
    #   'input_mask' : 1 at arguments for the current event else 0 (input_token_ids)
    #   'compare_mask': 1 at arguments for the current event else 0 (compare_token_ids)

    ontology_dict = load_ontology(dataset) 
    max_tokens = 0
    max_tgt =0 
    print(f"source:{source}")
    print(f"target:{target}")
    print(f'trigger dis:{trigger_dis}')
    print(f'dataset:{dataset}')

    with open(source, 'r') as reader , open(target, 'w') as writer:
        total_cnt = 0
        cnt = 0


        for line in tqdm(reader):
            ex = json.loads(line.strip())

            if 'entity_mentions' in ex:
                id2entity = {}
                for entity in ex["entity_mentions"]:
                    id2entity[entity["id"]] = entity
            else:
                id2entity = None
            
            event_range = {}
            for i in range(len(ex['event_mentions'])):
                # if len(ex['event_mentions'][i]['arguments']) > 0:
                start = ex['event_mentions'][i]["trigger"]['start']
                end =ex['event_mentions'][i]["trigger"]['end']
                event_range[i] = {'start':start,'end':end}
            events = event_range.keys()

            for i in range(len(ex['event_mentions'])):
                evt_type = ex['event_mentions'][i]['event_type']
                
                assert evt_type in ontology_dict 

                close_events = list(filter(lambda x:abs(event_range[x]['start'] - event_range[i]['start']) <= trigger_dis and x!=i, events)) # events whose triggers are close to the current trigger
                if len(close_events):
                    cnt += 1
                input_template, output_template, context_tag_trigger, context_tag_trigger_mask, context_tag_other_events, context_tag_other_events_mask = create_instance_tag_comparing(ex, ontology_dict, index=i, close_events=close_events, tokenizer = tokenizer, id2entity=id2entity)


                
                max_tokens = max(len(context_tag_trigger) + len(input_template) + 4, max_tokens)
                max_tgt = max(len(output_template) +1 , max_tgt)
                # print(len(context_tag_other_events), len(input_template), len(context_tag_trigger))
                # print(max_tokens, len(context_tag_trigger), len(input_template))
                # assert max_tokens <= MAX_LENGTH
                assert max_tgt <= MAX_TGT_LENGTH
                

                input_tokens = tokenizer.encode_plus(input_template,context_tag_trigger, 
                    add_special_tokens=True,
                    add_prefix_space=True,
                    max_length=MAX_LENGTH,
                    truncation='only_second',
                    padding='max_length')

                tgt_tokens = tokenizer.encode_plus(output_template, 
                    add_special_tokens=True,
                    add_prefix_space=True, 
                    max_length=MAX_TGT_LENGTH,
                    truncation=True,
                    padding='max_length')

                compare_tokens = tokenizer.encode_plus(input_template, context_tag_other_events, 
                    add_special_tokens=True,
                    add_prefix_space=True, 
                    max_length=MAX_LENGTH,
                    truncation='only_second',
                    padding='max_length')
                
                input_mask = [0] * (1 + len(input_template) + 2) + context_tag_trigger_mask + [0]*(MAX_LENGTH - 3 - len(context_tag_trigger_mask) - len(input_template))
                compare_mask = [0] * (1 + len(input_template) + 2) + context_tag_other_events_mask + [0]*(MAX_LENGTH - 3 - len(context_tag_other_events_mask) - len(input_template))
                input_mask = input_mask[:MAX_LENGTH]
                compare_mask = compare_mask[:MAX_LENGTH]
            
                processed_ex = {
                    'event_idx': i, 
                    'doc_key': ex['doc_id'], 
                    'input_token_ids':input_tokens['input_ids'],
                    'input_attn_mask': input_tokens['attention_mask'],
                    'tgt_token_ids': tgt_tokens['input_ids'],
                    'tgt_attn_mask': tgt_tokens['attention_mask'],
                    'compare_token_ids': compare_tokens['input_ids'],
                    'compare_attn_mask': compare_tokens['attention_mask'],
                    'input_mask' : input_mask,
                    'compare_mask':compare_mask
                }
                
                # if cnt < 5:
                #     print(f"example {total_cnt}")
                #     tokens = tokenizer.convert_ids_to_tokens(processed_ex["input_token_ids"])
                #     for i,_ in enumerate(zip(tokens, processed_ex['input_mask'])):
                #         print(i,_)
                #     tokens = tokenizer.convert_ids_to_tokens(processed_ex["compare_token_ids"])
                #     for i,_ in enumerate(zip(tokens, processed_ex['compare_mask'])):
                #         print(i,_)
                #     print('-'*80)
                
                
                writer.write(json.dumps(processed_ex) + '\n')
                total_cnt += 1
    print(f'total_event: {total_cnt}')
    print(f'has close events: {cnt}')

    print('longest context:{}'.format(max_tokens))
    print('longest target {}'.format(max_tgt))