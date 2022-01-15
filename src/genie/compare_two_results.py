import os 
import json 
import argparse 
import re 
from copy import deepcopy
from collections import defaultdict 
from tqdm import tqdm
import spacy 
from transformers import BartTokenizer


from utils import load_ontology,find_arg_span, compute_f1, get_entity_span, find_head, WhitespaceTokenizer

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

'''
Scorer for argument extraction on ACE & KAIROS.
For the RAMS dataset, the official scorer is used. 

Outputs: 
Head F1 
Coref F1 
'''
def clean_span(ex, span):
    tokens = ex['tokens']
    if tokens[span[0]].lower() in {'the', 'an', 'a'}:
        if span[0]!=span[1]:
            return (span[0]+1, span[1])
    return span 

def extract_args_from_template(ex, template, ontology_dict,):
    # extract argument text 
    template_words = template.strip().split()
    predicted_words = ex['predicted'].strip().split()    
    predicted_args = defaultdict(list) # each argname may have multiple participants 
    t_ptr= 0
    p_ptr= 0 
    evt_type = ex['event']['event_type']
    while t_ptr < len(template_words) and p_ptr < len(predicted_words):
        if re.match(r'<(arg\d+)>', template_words[t_ptr]):
            m = re.match(r'<(arg\d+)>', template_words[t_ptr])
            arg_num = m.group(1)
            try:
                arg_name = ontology_dict[evt_type][arg_num]
            except KeyError:
                print(evt_type)
                exit() 

            if predicted_words[p_ptr] == '<arg>':
                # missing argument
                p_ptr +=1 
                t_ptr +=1  
            else:
                arg_start = p_ptr 
                while (p_ptr < len(predicted_words)) and ((t_ptr== len(template_words)-1) or (predicted_words[p_ptr] != template_words[t_ptr+1])):
                    p_ptr+=1 
                arg_text = predicted_words[arg_start:p_ptr]
                predicted_args[arg_name].append(arg_text)
                t_ptr+=1 
                # aligned 
        else:
            t_ptr+=1 
            p_ptr+=1 
    
    return predicted_args



def scorer(args):
    ontology_dict = load_ontology(dataset=args.dataset)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    tokenizer.add_tokens([' <arg>',' <tgr>', ' <tag>', ' </tag>'])

    coref_mapping = defaultdict(dict) # span to canonical entity_id mapping for each doc 
    if args.coref:
        if args.dataset == 'KAIROS' and args.coref_file:
            with open(args.coref_file, 'r') as f, open(args.test_file, 'r') as test_reader:
                for line, test_line  in zip(f, test_reader):
                    coref_ex = json.loads(line)
                    ex = json.loads(test_line)
                    doc_id = coref_ex['doc_key']
                    
                    for cluster, name in zip(coref_ex['clusters'], coref_ex['informative_mentions']):
                        canonical = cluster[0]
                        for ent_id in cluster:
                            ent_span = get_entity_span(ex, ent_id) 
                            ent_span = (ent_span[0], ent_span[1]-1) 
                            coref_mapping[doc_id][ent_span] = canonical
                    # this does not include singleton clusters 
        else:
            # for the ACE dataset 
            with open(args.test_file) as f:
                for line in f:
                    doc=json.loads(line.strip())
                    doc_id = doc['sent_id']
                    for entity in doc['entity_mentions']:
                        mention_id = entity['id']
                        ent_id = '-'.join(mention_id.split('-')[:-1]) 
                        coref_mapping[doc_id][(entity['start'], entity['end']-1)] = ent_id # all indexes are inclusive 

    def get_examples(gen_file, input_file):
        examples = {}
        doc2ex = defaultdict(list) # a document contains multiple events 
        with open(gen_file,'r') as gen, open(input_file, 'r') as input_f:
            for lidx, (gen_line, input_line)  in enumerate(zip(gen, input_f)): # this solution relies on keeping the exact same order 
                pred = json.loads(gen_line.strip())
                input_ids = json.loads(input_line.strip())
                examples[lidx] = {
                    'predicted': pred['predicted'],
                    'gold': pred['gold'],
                    'doc_id': pred['doc_key'],
                    'input_ids': input_ids['input_token_ids']
                }
                doc2ex[pred['doc_key']].append(lidx)

        with open(args.test_file, 'r') as f:
            for line in f:
                doc = json.loads(line.strip())
                if 'sent_id' in doc.keys():
                    doc_id = doc['sent_id']
                    # print('evaluating on sentence level')
                else:
                    doc_id = doc['doc_id']
                    # print('evaluating on document level')
                for idx, eid in enumerate(doc2ex[doc_id]):
                    examples[eid]['tokens'] = doc['tokens']
                    examples[eid]['event'] = doc['event_mentions'][idx]
                    examples[eid]['entity_mentions'] = doc['entity_mentions']    
        return examples
    
    examples_good = get_examples(args.gen_good_file, args.input_good)
    examples_bad = get_examples(args.gen_bad_file, args.input_bad)

    doc_events = defaultdict(list)
    items_good = list(examples_good.items())
    items_bad = list(examples_bad.items())
    for (key_good, ex_good), (key_bad, ex_bad) in tqdm(zip(items_good, items_bad)):
        assert key_good == key_bad

        context_words = ex_good['tokens']
        doc_id = ex_good['doc_id']
        doc = None 
        if args.head_only:
            doc = nlp(' '.join(context_words))
        evt_type = ex_good['event']['event_type']
        assert evt_type in ontology_dict
        template = ontology_dict[evt_type]['template']
        
        
        predicted_args_good = extract_args_from_template(ex_good,template, ontology_dict)
        predicted_args_bad = extract_args_from_template(ex_bad,template, ontology_dict)
        # {'Victim': [['members']], 'Killer': [['Taliban']]}

        # get trigger 
        # extract argument span
        trigger_start = ex_good['event']['trigger']['start']
        trigger_end = ex_good['event']['trigger']['end']

        gold_set = set() 
        event_arg = {}
        gold_canonical_set = set() # set of canonical mention ids, singleton mentions will not be here 
        for arg in ex_good['event']['arguments']:
            argname = arg['role']
            entity_id = arg['entity_id']
            event_arg[entity_id] = {"role":argname,"predict":False}
            span = get_entity_span(ex_good, entity_id)
            span = (span[0], span[1]-1)
            span = clean_span(ex_good, span)
            # clean up span by removing `a` `the`
            if args.head_only and span[0]!=span[1]:
                span = find_head(span[0], span[1], doc=doc) 
            
            gold_set.add((span[0], span[1], evt_type, entity_id, argname))
            if args.coref:
                if span in coref_mapping[doc_id]:
                    canonical_id = coref_mapping[doc_id][span]
                    gold_canonical_set.add((canonical_id, evt_type, argname))
        
        def get_set(predicted_args):
            # (start, end, evt_type, argname)
            predicted_set = set() 
            for argname in predicted_args:
                for entity in predicted_args[argname]:# this argument span is inclusive, FIXME: this might be problematic 
                    arg_span = find_arg_span(entity, context_words, 
                        trigger_start, trigger_end, head_only=args.head_only, doc=doc) 
                    # print(entity, arg_span)
                    # ['members'] (6, 6)
                    
                    if arg_span:# if None means hullucination
                        
                        predicted_set.add((arg_span[0], arg_span[1], evt_type, argname))

                    else:
                        new_entity = []
                        for w in entity:
                            if w == 'and' and len(new_entity) >0:
                                arg_span = find_arg_span(new_entity, context_words, trigger_start, trigger_end,
                                head_only=args.head_only, doc=doc)
                                if arg_span: predicted_set.add((arg_span[0], arg_span[1], evt_type, argname))
                                new_entity = []
                            else:
                                new_entity.append(w)
                        
                        if len(new_entity) >0: # last entity
                            arg_span = find_arg_span(new_entity, context_words, trigger_start, trigger_end, 
                            head_only=args.head_only, doc=doc)
                            if arg_span: predicted_set.add((arg_span[0], arg_span[1], evt_type, argname))
                
            return predicted_set
        
        predicted_set_good = get_set(predicted_args_good)
        predicted_set_bad = get_set(predicted_args_bad)

        def get_num(predicted_set):
            cnt = 0
            for pred_arg in predicted_set:
                arg_start, arg_end, event_type, role = pred_arg
                gold_idn = {item for item in gold_set
                            if item[0] == arg_start and item[1] == arg_end
                            and item[2] == event_type}
                if gold_idn:
                    cnt += 1
            return cnt

        def get_arg_text(arg_set):
            arg_set_with_text = []
            for arg in arg_set:
                arg_text = ' '.join(context_words[arg[0]:arg[1] + 1])
                arg_set_with_text.append((arg, arg_text))
            return arg_set_with_text
        
        cnt  = get_num(predicted_set_good)
        # if cnt != len(predicted_set_good) or cnt != len(gold_set):
        if get_num(predicted_set_good) > get_num(predicted_set_bad):
            print(f"input_good: {tokenizer.decode(ex_good['input_ids'], skip_special_tokens=True)}")
            print(f"input_bad: {tokenizer.decode(ex_bad['input_ids'], skip_special_tokens=True)}")
            print(f"template_good:{ex_good['predicted']}")
            print(f"template_bad:{ex_bad['predicted']}")
            print(f"template_gold:{ex_bad['gold']}")
            print('predicted_good:')
            for arg in get_arg_text(predicted_set_good):
                print(arg)
            print('predicted_bad:')
            for arg in get_arg_text(predicted_set_bad):
                print(arg)
            print('gold:')
            for arg in get_arg_text(gold_set):
                print(arg)
            print(f"trigger_start: {trigger_start}, trigger_end: {trigger_end}")
            print('-'*80)  

        # types = set(arg[4] for arg in gold_set)
        # if len(types) >= 4:
        #     print(f"input_good: {tokenizer.decode(ex_good['input_ids'], skip_special_tokens=True)}")
        #     print(f"input_bad: {tokenizer.decode(ex_bad['input_ids'], skip_special_tokens=True)}")
        #     print(f"predicted_good: {predicted_set_good}")
        #     print(f"predicted_bad: {predicted_set_bad}")
        #     print(f"gold: {gold_set}")
        #     print(f"trigger_start: {trigger_start}, trigger_end: {trigger_end}")
        #     print('-'*80)

        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_good_file',type=str,default=f'checkpoints/debug_iter/epoch_5_test_step_1_predictions.jsonl' )
    # parser.add_argument('--gen_good_file',type=str,default=f'checkpoints/siterative_fast_5e_5_40_info/epoch_5_test_step_0_predictions.jsonl' )
    # parser.add_argument('--gen_bad_file', type=str,default='checkpoints/s_12-pred/predictions.jsonl')
    parser.add_argument('--gen_bad_file', type=str,default='checkpoints/debug_iter/epoch_5_test_step_0_predictions.jsonl')
    # parser.add_argument('--gen_bad_file', type=str,default='checkpoints/iterative_fast_5e_5_40_info/epoch_5_test_step_2_predictions.jsonl')

    parser.add_argument('--input_good',type=str,default=f'preprocessed/preprocessed_debug_iter/test_step_0_data.json' )
    # parser.add_argument('--input_good',type=str,default=f'/test.jsonl' )
    parser.add_argument('--input_bad',type=str,default=f'preprocessed_KAIROS/test.jsonl' )
    # parser.add_argument('--input_bad',type=str,default=f'preprocessed_iterative_fast_5e_5_40_info/test_step_1_data.json' )

    parser.add_argument('--test-file', type=str,default='data/wikievents/test_no_ontology.jsonl')
    # parser.add_argument('--test-file', type=str,default='data/wikievents/test_info_no_ontology.jsonl')

    parser.add_argument('--coref-file', type=str,default='data/wikievents/coref/test.jsonlines')
    parser.add_argument('--head-only', action='store_true',default=True)
    parser.add_argument('--coref', action='store_true',default=True)
    parser.add_argument('--dataset',type=str, default='KAIROS', choices=['ACE', 'KAIROS','AIDA'])
    arg_scorer = parser.parse_args() 

    scorer(arg_scorer)


    

                    




