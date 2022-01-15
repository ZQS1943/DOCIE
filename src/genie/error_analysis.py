from logging import error
import os 
import json 
import argparse 
import re 
from copy import deepcopy
from collections import defaultdict
from spacy.util import resolve_dot_names 
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
    if 'scores' in ex:
        scores = ex['scores']
        for i in range(len(scores), len(predicted_words)):
            if '<arg>' in predicted_words[i]:
                scores.append(0.0)
            else:
                scores.append(1.0)

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
                # if arg_start == p_ptr:
                if predicted_words[arg_start: arg_start + 4] == ['explosive', 'device', 'explosive', 'device']:
                    p_ptr += 2
                elif predicted_words[arg_start: arg_start + 3] == ['explosive', 'explosive', 'device']:
                    p_ptr += 1
                elif predicted_words[arg_start: arg_start + 2] == ['court', 'court']:
                    p_ptr += 1
                elif predicted_words[arg_start: arg_start + 2] == ['explosive', 'devices']:
                    p_ptr += 2

                while (p_ptr < len(predicted_words)) and ((t_ptr== len(template_words)-1) or (predicted_words[p_ptr] != template_words[t_ptr+1])):
                    p_ptr+=1
                arg_text = predicted_words[arg_start:p_ptr]
                # print(predicted_words)
                # print(template_words)
                # print(ex['predicted'])
                if 'scores' in ex:  
                    arg_score = sum(ex['scores'][arg_start:p_ptr])/(p_ptr - arg_start) if p_ptr - arg_start else 0
                # arg_score = sum(ex['scores'][arg_start:p_ptr])/(p_ptr - arg_start)
                else:
                    arg_score = 1
                
                predicted_args[arg_name].append((arg_text, arg_score))
                t_ptr+=1 
                # aligned 
        else:
            t_ptr+=1 
            p_ptr+=1 
    # if "explosive Instrument Instrument Instrument Instrument Instrumentation" in ex['predicted']:
    # print(predicted_args)
    # assert 1==0
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
    cnt_no_trigger  = 0
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
                    'input_ids': input_ids['input_token_ids'],
                    'scores': pred['scores']
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
    
    def get_score_for_one_event(ex):
        pred_arg_num =0 
        gold_arg_num =0
        arg_idn_num =0 
        arg_class_num =0 

        arg_idn_coref_num =0
        arg_class_coref_num =0

        context_words = ex['tokens']
        doc_id = ex['doc_id']
        doc = None 
        if args.head_only:
            doc = nlp(' '.join(context_words))

        predicted_result = {}
        
        # get template 
        evt_type = ex['event']['event_type']
        template = ontology_dict[evt_type]['template']
        # extract argument text 
        predicted_args = extract_args_from_template(ex,template, ontology_dict)
        # get trigger 
        # extract argument span
        trigger_start = ex['event']['trigger']['start']
        trigger_end = ex['event']['trigger']['end']
        
        predicted_set = set() 
        for argname in predicted_args:
            for entity, score in predicted_args[argname]:# this argument span is inclusive, FIXME: this might be problematic 
                predicted_result[argname] = {
                    'predicted_text': entity,
                    'score': score,
                    'extracted_args': [],
                    'gold': []
                }
                arg_span = find_arg_span(entity, context_words, 
                    trigger_start, trigger_end, head_only=args.head_only, doc=doc) 
                
                new_entity = []
                for w in entity:
                    if w == 'and' and len(new_entity) >0:
                        arg_span = find_arg_span(new_entity, context_words, trigger_start, trigger_end,
                        head_only=args.head_only, doc=doc)
                        if arg_span: predicted_set.add((arg_span[0], arg_span[1], evt_type, argname, score))
                        new_entity = []
                    else:
                        new_entity.append(w)
                
                if len(new_entity) >0: # last entity
                    arg_span = find_arg_span(new_entity, context_words, trigger_start, trigger_end, 
                    head_only=args.head_only, doc=doc)
                    if arg_span: predicted_set.add((arg_span[0], arg_span[1], evt_type, argname, score))
                    
                   
                    
        # get gold spans         
        gold_set = set() 
        event_arg = {}
        gold_canonical_set = set() # set of canonical mention ids, singleton mentions will not be here 
        
        for arg in ex['event']['arguments']:
            argname = arg['role']
            entity_id = arg['entity_id']
            event_arg[entity_id] = {"role":argname,"predict":False}
            span = get_entity_span(ex, entity_id)
            span = (span[0], span[1]-1)
            span = clean_span(ex, span)
            predicted_args[argname].append(ex['tokens'][span[0]:span[1] + 1])
            # clean up span by removing `a` `the`
            if args.head_only and span[0]!=span[1]:
                span = find_head(span[0], span[1], doc=doc) 
            
            gold_set.add((span[0], span[1], evt_type, entity_id, argname))

            if argname not in predicted_result:
                predicted_result[argname] = {}
                predicted_result[argname]['gold'] = []
            predicted_result[argname]['gold'].append(ex['tokens'][span[0]:span[1] + 1])
            
            if args.coref:
                if span in coref_mapping[doc_id]:
                    canonical_id = coref_mapping[doc_id][span]
                    gold_canonical_set.add((canonical_id, evt_type, argname))
        # print(ex['event']['arguments'])
        # print(predicted_set)

        pred_arg_num += len(predicted_set)
        gold_arg_num += len(gold_set)
        # check matches 
        arg_idn_num_tmp = 0
        arg_class_num_tmp =0 

        arg_idn_coref_num_tmp =0
        arg_class_coref_num_tmp =0
        for pred_arg in predicted_set:
            arg_start, arg_end, event_type, role, score = pred_arg
            predicted_result[role]['extracted_args'].append(ex['tokens'][arg_start: arg_end + 1])
            gold_idn = {item for item in gold_set
                        if item[0] == arg_start and item[1] == arg_end
                        and item[2] == event_type}
            if gold_idn:
                arg_idn_num += 1
                arg_idn_num_tmp += 1
                gold_class = {item for item in gold_idn if item[-1] == role}
                if gold_class:
                    arg_class_num += 1
                    arg_class_num_tmp += 1
                    for item in gold_class:
                        event_arg[item[3]]["predict"] = True
            elif args.coref:# check coref matches 
                arg_start, arg_end, event_type, role, score = pred_arg
                span = (arg_start, arg_end)
                if span in coref_mapping[doc_id]:
                    canonical_id = coref_mapping[doc_id][span]
                    gold_idn_coref = {item for item in gold_canonical_set 
                        if item[0] == canonical_id and item[1] == event_type}
                    if gold_idn_coref:
                        arg_idn_coref_num +=1 
                        arg_idn_coref_num_tmp +=1 
                        gold_class_coref = {item for item in gold_idn_coref
                        if item[2] == role}
                        if gold_class_coref:
                            arg_class_coref_num +=1 
                            arg_class_coref_num_tmp +=1 
        # print(pred_arg_num, gold_arg_num, arg_idn_num)
        # print(pred_arg_num, gold_arg_num, arg_class_num)
        role_id_prec, role_id_rec, role_id_f = compute_f1(
            pred_arg_num, gold_arg_num, arg_idn_num)
        role_prec, role_rec, role_f = compute_f1(
            pred_arg_num, gold_arg_num, arg_class_num)

        corefrole_id_prec, corefrole_id_rec, corefrole_id_f = compute_f1(
            pred_arg_num, gold_arg_num, arg_idn_num + arg_idn_coref_num)
        corefrole_prec, corefrole_rec, corefrole_f = compute_f1(
            pred_arg_num, gold_arg_num, arg_class_num + arg_class_coref_num)

        return (role_id_f, role_f, corefrole_id_f, corefrole_f), predicted_result
    
    examples = get_examples(args.gen_file, args.input_file)

    doc_events = defaultdict(list)
    items = list(examples.items())


    error_cases = []

    print(f'event_id\tinput\tScores\tRole\tPredicted\tGold')
    for i, (key, ex) in tqdm(enumerate(items)):
        # if i != 197:
        #     continue
        if "<tgr>" not in tokenizer.decode(ex['input_ids'], skip_special_tokens=True):
            cnt_no_trigger += 1
        scores, predicted_args = get_score_for_one_event(ex)
        # for role in predicted_args:
        #     print(role, json.dumps(predicted_args[role], indent=True))
        if sum(scores) != 4.0:
            error_case = {}
            if len(predicted_args) == 0:
                continue
            
            # print(f"{i}\t{tokenizer.decode(ex['input_ids'], skip_special_tokens=True)}\t{scores}\t\t\t")
            input = tokenizer.decode(ex['input_ids'], skip_special_tokens=True)
            input_tokens = re.findall(r"[\w']+|[.,!?;]|<tgr>|<arg>|<tag>|</tag>", input)
            # print(input_tokens)
            error_case['event_id'] = i
            error_case['input_tokens'] = input_tokens
            error_case['event_type'] = ex['event']['event_type']
            
            trigger_start = input_tokens.index('<tgr>') + 1
            trigger_end = input_tokens.index('<tgr>', trigger_start)
            error_case['trigger'] = {
                'start':trigger_start,
                'end':trigger_end
            }
            error_case['Coref Role score'] = scores[3]
            error_case['predicted'] = ex['predicted']
            # print(input_tokens[trigger_start:trigger_end])
            # print(input_tokens)
            # print(predicted_args)
            # assert 1==0
            arguments = defaultdict(list)
            for role in predicted_args:
                
                # get_visualize
                if 'extracted_args' in predicted_args[role]:
                    for arg in predicted_args[role]["extracted_args"]:
                        span = find_arg_span(arg, input_tokens, trigger_start, trigger_end)
                        # print('entity:',arg)
                        # print(ex['tokens'])
                        # print(input_tokens)
                        # assert len(span) == 2
                        arguments[(span,role)].append('predicted')
                
                if 'gold' in predicted_args[role]:
                    for arg in predicted_args[role]["gold"]:
                        span = find_arg_span(arg, input_tokens, trigger_start, trigger_end)
                        assert len(span) == 2
                        arguments[(span,role)].append('gold')
            error_case['arguments'] = []

            for arg in arguments:
                types = arguments[arg]
                assert types == ['predicted'] or types == ['gold'] or types == ['gold','predicted'] or types == ['predicted','gold']
                error_case['arguments'].append({
                    'start': arg[0][0],
                    'end': arg[0][1] + 1,
                    'role': arg[1],
                    'type': types[0] if len(types) == 1 else 'both'
                })
            
            error_cases.append(error_case)
            

                # print('\t\t\t'+role, end='\t')
                # if 'extracted_args' in predicted_args[role]:
                #     arg_list = []
                #     for arg in predicted_args[role]["extracted_args"]:
                #         arg_list.append(" ".join(arg))
                #     print(';'.join(arg_list),end='')
                # print('\t', end='')
                # if 'gold' in predicted_args[role]:
                #     print(' '.join(predicted_args[role]['gold']), end='')
                # print('\t')  

            # print('predicted_good:')
            # for arg in get_arg_text(predicted_set_good):
            #     print(arg)
            # print('predicted_bad:')
            # for arg in get_arg_text(predicted_set_bad):
            #     print(arg)
            # print('gold:')
            # for arg in get_arg_text(gold_set):
            #     print(arg)
            # print(f"trigger_start: {trigger_start}, trigger_end: {trigger_end}")
            # print('-'*80)  

        # types = set(arg[4] for arg in gold_set)
        # if len(types) >= 4:
        #     print(f"input_good: {tokenizer.decode(ex_good['input_ids'], skip_special_tokens=True)}")
        #     print(f"input_bad: {tokenizer.decode(ex_bad['input_ids'], skip_special_tokens=True)}")
        #     print(f"predicted_good: {predicted_set_good}")
        #     print(f"predicted_bad: {predicted_set_bad}")
        #     print(f"gold: {gold_set}")
        #     print(f"trigger_start: {trigger_start}, trigger_end: {trigger_end}")
        #     print('-'*80)
    # print(f'cnt_no_trigger:{cnt_no_trigger}')
    with open(args.gen_file.replace('predictions.jsonl', 'error_cases.json'), 'w') as f:
        f.write(json.dumps(error_cases, indent=True))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-file',type=str,default=f'checkpoints/comparing_40_0.5_9/epoch_5_test_step_4_predictions.jsonl' )
    parser.add_argument('--input_file',type=str,default=f'preprocessed/preprocessed_comparing_40_0.5_9/test_step_3_data.json' )

    # parser.add_argument('--input_bad',type=str,default=f'preprocessed_iter_sentence_selection_3e-5/test.json' )
    # parser.add_argument('--input_bad',type=str,default=f'preprocessed_sentence_selection_nb/test_data.jsonl' )
    # parser.add_argument('--input_bad',type=str,default=f'preprocessed_iterative_fast_5e_5_40_info/test_step_1_data.json' )

    parser.add_argument('--test-file', type=str,default='data/wikievents/test_no_ontology.jsonl')
    # parser.add_argument('--test-file', type=str,default='data/wikievents/test_info_no_ontology.jsonl')

    parser.add_argument('--coref-file', type=str,default='data/wikievents/coref/test.jsonlines')
    parser.add_argument('--head-only', action='store_true',default=True)
    parser.add_argument('--coref', action='store_true',default=True)
    parser.add_argument('--dataset',type=str, default='KAIROS', choices=['ACE', 'KAIROS','AIDA'])
    arg_scorer = parser.parse_args() 

    scorer(arg_scorer)


    

                    




