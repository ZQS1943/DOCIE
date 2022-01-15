import os 
import json 
import argparse 
import re 
from copy import deepcopy
from collections import defaultdict 
from tqdm import tqdm
import spacy 
import numpy as np

from .utils import load_ontology,find_arg_span_original_text, compute_f1, get_entity_span, find_head, WhitespaceTokenizer

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

def scorer_bert_crf(args):
    print(f"gen_file:{args.gen_file}")
    print(f"test_file:{args.test_file}")
    # print(f'coref_file:{args.coref_file}')
    ontology_dict = load_ontology(dataset=args.dataset)

    docid2doc = {}

    examples = {}
    doc2ex = defaultdict(list) # a document contains multiple events 
    with open(args.gen_file,'r') as f:
        for lidx, line in enumerate(f): # this solution relies on keeping the exact same order 
            pred = json.loads(line.strip()) 
            examples[lidx] = {
                'predicted': pred['predicted'],
                'gold': pred['gold'],
                'doc_id': pred['doc_key'],
            }
            if 'scores' in pred:
                examples[lidx]['scores'] = pred['scores']
            doc2ex[pred['doc_key']].append(lidx)
            

    with open(args.test_file, 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            docid2doc[doc['doc_id']] = {
                "doc_id": doc["doc_id"],
                "tokens": doc["tokens"],
                "sentences": doc["sentences"],
                'event_mentions': [],
                'entity_mentions':doc['entity_mentions']
            }
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
                    doc_id = doc['doc_id']
                    for idx,cluster in enumerate(doc['coref_entities']):
                        for start, end in cluster:
                            coref_mapping[doc_id][(start, end)] = idx
    ic_cc = []
    ic_cf = []
    if_ = [] 

    pred_arg_num =0 
    gold_arg_num =0
    arg_idn_num =0 
    arg_class_num =0 

    arg_idn_coref_num =0
    arg_class_coref_num =0

    event_idn_num =0
    event_class_num =0

    doc_events = defaultdict(list)
    items = list(examples.items())
    for key,ex in tqdm(items):
        # if key != 138:
        #     continue
        context_words = ex['tokens']
        doc_id = ex['doc_id']
        doc = None 
        if args.head_only:
            doc = nlp(' '.join(context_words))
        
        # get template 
        evt_type = ex['event']['event_type']

        assert evt_type in ontology_dict
        
        template = ontology_dict[evt_type]['template']
        # extract argument text 
        predicted_args = defaultdict(list)
        for arg in ex['predicted']:
            predicted_args[arg[1]].append(arg[0])
        # {'Victim': [['members']], 'Killer': [['Taliban']]}

        # get trigger 
        # extract argument span
        trigger_start = ex['event']['trigger']['start']
        trigger_end = ex['event']['trigger']['end']

        
        
        # (start, end, evt_type, argname)
        predicted_set = set() 
        predicted_set_original_text = set()
        for argname in predicted_args:
            for entity in predicted_args[argname]:# this argument span is inclusive, FIXME: this might be problematic 
                score = 0
                if len(entity) == 0:
                    continue
                # arg_span_original_text, arg_span = find_arg_span_original_text(entity, context_words, 
                #     trigger_start, trigger_end, head_only=args.head_only, doc=doc) 
                
                # # ['members'] (6, 6)
                
                # if arg_span:# if None means hullucination
                    
                #     predicted_set.add((arg_span[0], arg_span[1], evt_type, argname, score))
                #     predicted_set_original_text.add((arg_span_original_text[0], arg_span_original_text[1], evt_type, argname, score))

                # else:
                new_entity = []
                for w in entity:
                    if w == 'and' and len(new_entity) >0:
                        arg_span_original_text, arg_span = find_arg_span_original_text(new_entity, context_words, trigger_start, trigger_end,
                        head_only=args.head_only, doc=doc)
                        if arg_span: 
                            predicted_set.add((arg_span[0], arg_span[1], evt_type, argname, score))
                            predicted_set_original_text.add((arg_span_original_text[0], arg_span_original_text[1], evt_type, argname, score))
                        new_entity = []
                    else:
                        new_entity.append(w)
                
                if len(new_entity) >0: # last entity
                    arg_span_original_text, arg_span = find_arg_span_original_text(new_entity, context_words, trigger_start, trigger_end, 
                    head_only=args.head_only, doc=doc)
                    if arg_span: 
                        predicted_set.add((arg_span[0], arg_span[1], evt_type, argname, score))
                        predicted_set_original_text.add((arg_span_original_text[0], arg_span_original_text[1], evt_type, argname, score))
        
        arguments = []
        for start, end, _, argname, score in predicted_set_original_text:
            # print(context_words[start:end + 1], argname)
            if score > args.score_th:
                arguments.append({
                    "start":start,
                    "end": end + 1,
                    "role": argname,
                    "text": " ".join(context_words[start:end + 1]),
                    "score": score
                })
        docid2doc[doc_id]["event_mentions"].append({
            "event_type": evt_type,
            "trigger": ex['event']['trigger'],
            "arguments": arguments
        })
        # assert 1==0
                    
        # get gold spans 
        # (span[0], span[1], evt_type, entity_id, argname)        
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
            # clean up span by removing `a` `the`
            if args.head_only and span[0]!=span[1]:
                span = find_head(span[0], span[1], doc=doc) 
            
            gold_set.add((span[0], span[1], evt_type, entity_id, argname))
            if args.coref:
                if span in coref_mapping[doc_id]:
                    canonical_id = coref_mapping[doc_id][span]
                    gold_canonical_set.add((canonical_id, evt_type, argname))
        
        
        pred_arg_num += len(predicted_set)
        gold_arg_num += len(gold_set)
        # check matches 
        arg_idn_num_tmp = 0
        arg_class_num_tmp =0 

        arg_idn_coref_num_tmp =0
        arg_class_coref_num_tmp =0
        for pred_arg in predicted_set:
            arg_start, arg_end, event_type, role, score = pred_arg
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
                    ic_cc.append(score)
                else:
                    ic_cf.append(score)
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
                            ic_cc.append(score)
                        else:
                            ic_cf.append(score) 
                    else:
                        if_.append(score)

        # print(predicted_set)
        # print(gold_set)
        # print(f'''arg_idn_num_tmp = {arg_idn_num_tmp},
        # arg_class_num_tmp = {arg_class_num_tmp},
        # arg_idn_coref_num_tmp = {arg_idn_coref_num_tmp},
        # arg_class_coref_num_tmp = {arg_class_coref_num_tmp}''')
        examples[key]["arg_class_coref_num_tmp"] = arg_class_coref_num_tmp
        examples[key]["exact_match"] = False
        if len(gold_set) == arg_class_num_tmp + arg_class_coref_num_tmp:
            event_idn_num += 1
            event_class_num += 1
            examples[key]["exact_match"] = True

        doc_events[doc_id].append(event_arg)

    
    pair_arg = []
    for doc_id in doc_events:
        events = doc_events[doc_id]
        for ei in range(len(events) - 1):
            for ej in range(ei + 1, len(events)):
                for entity_id in events[ei]:
                    if entity_id in events[ej]:
                        pair_arg.append((events[ei][entity_id], events[ej][entity_id]))
    # for item in pair_arg:
    #     print(item)
    pair_arg_inconsistent = [1 if x[0]['predict'] != x[1]['predict'] else 0 for x in pair_arg]
    if len(pair_arg):
        print(f'paired args: {len(pair_arg)}, inconsistent args: {sum(pair_arg_inconsistent)}, raito: {sum(pair_arg_inconsistent)/len(pair_arg)}')      
        
    if args.head_only:
        print('Evaluation by matching head words only....')
    
    
    role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_class_num)

    print("Event-level identification: P: {:.2f}".format(event_idn_num/len(examples)*100.0))
    print("Event-level : P: {:.2f}".format(event_class_num/len(examples)*100.0))

    print(f"gold arg num: {gold_arg_num}")    
    print('Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    print('Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_prec * 100.0, role_rec * 100.0, role_f * 100.0))

    if args.coref:
        role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num + arg_idn_coref_num)
        role_prec, role_rec, role_f = compute_f1(
            pred_arg_num, gold_arg_num, arg_class_num + arg_class_coref_num)

        
        print('Coref Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
            role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
        print('Coref Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
            role_prec * 100.0, role_rec * 100.0, role_f * 100.0))

        print('len:',len(ic_cc), len(ic_cf), len(if_))
        print('mean:',np.mean(ic_cc), np.mean(ic_cf), np.mean(if_))
        print('var:',np.var(ic_cc), np.var(ic_cf), np.var(if_))
        print('min:', min(ic_cc), min(ic_cf), min(if_))
        print('max:', max(ic_cc), max(ic_cf), max(if_))

    with open(args.gen_file.replace("predictions","eval_results")[:-1],'w') as f:
        f.write(json.dumps(examples,indent=True))   

    with open(args.gen_file.replace("predictions","results_for_predict")[:-1],'w') as f:
        for docid in docid2doc:
            f.write(json.dumps(docid2doc[docid]) + '\n')    
    

def scorer(args):
    print(f"gen_file:{args.gen_file}")
    print(f"test_file:{args.test_file}")
    # print(f'coref_file:{args.coref_file}')
    ontology_dict = load_ontology(dataset=args.dataset)

    if args.dataset == 'KAIROS' and args.coref and not args.coref_file:
        print('coreference file needed for the KAIROS dataset.')
        raise ValueError


    docid2doc = {}

    examples = {}
    doc2ex = defaultdict(list) # a document contains multiple events 
    with open(args.gen_file,'r') as f:
        for lidx, line in enumerate(f): # this solution relies on keeping the exact same order 
            pred = json.loads(line.strip()) 
            examples[lidx] = {
                'predicted': pred['predicted'],
                'gold': pred['gold'],
                'doc_id': pred['doc_key'],
            }
            if 'scores' in pred:
                examples[lidx]['scores'] = pred['scores']
            doc2ex[pred['doc_key']].append(lidx)
            

    with open(args.test_file, 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            docid2doc[doc['doc_id']] = {
                "doc_id": doc["doc_id"],
                "tokens": doc["tokens"],
                "sentences": doc["sentences"],
                'event_mentions': [],
                'entity_mentions':doc['entity_mentions']
            }
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
                    doc_id = doc['doc_id']
                    for idx,cluster in enumerate(doc['coref_entities']):
                        for start, end in cluster:
                            coref_mapping[doc_id][(start, end)] = idx

    ic_cc = []
    ic_cf = []
    if_ = [] 

    pred_arg_num =0 
    gold_arg_num =0
    arg_idn_num =0 
    arg_class_num =0 

    arg_idn_coref_num =0
    arg_class_coref_num =0

    event_idn_num =0
    event_class_num =0

    doc_events = defaultdict(list)
    items = list(examples.items())
    for key,ex in tqdm(items):
        # if key != 138:
        #     continue
        context_words = ex['tokens']
        doc_id = ex['doc_id']
        doc = None 
        if args.head_only:
            doc = nlp(' '.join(context_words))
        
        # get template 
        evt_type = ex['event']['event_type']

        assert evt_type in ontology_dict
        
        template = ontology_dict[evt_type]['template']
        # extract argument text 
        predicted_args = extract_args_from_template(ex,template, ontology_dict)
        # {'Victim': [['members']], 'Killer': [['Taliban']]}

        # get trigger 
        # extract argument span
        trigger_start = ex['event']['trigger']['start']
        trigger_end = ex['event']['trigger']['end']

        
        
        # (start, end, evt_type, argname)
        predicted_set = set() 
        predicted_set_original_text = set()
        for argname in predicted_args:
            for entity, score in predicted_args[argname]:# this argument span is inclusive, FIXME: this might be problematic 
                if len(entity) == 0:
                    continue
                # arg_span_original_text, arg_span = find_arg_span_original_text(entity, context_words, 
                #     trigger_start, trigger_end, head_only=args.head_only, doc=doc) 
                
                # # ['members'] (6, 6)
                
                # if arg_span:# if None means hullucination
                    
                #     predicted_set.add((arg_span[0], arg_span[1], evt_type, argname, score))
                #     predicted_set_original_text.add((arg_span_original_text[0], arg_span_original_text[1], evt_type, argname, score))

                # else:
                new_entity = []
                for w in entity:
                    if w == 'and' and len(new_entity) >0:
                        arg_span_original_text, arg_span = find_arg_span_original_text(new_entity, context_words, trigger_start, trigger_end,
                        head_only=args.head_only, doc=doc)
                        if arg_span: 
                            predicted_set.add((arg_span[0], arg_span[1], evt_type, argname, score))
                            predicted_set_original_text.add((arg_span_original_text[0], arg_span_original_text[1], evt_type, argname, score))
                        new_entity = []
                    else:
                        new_entity.append(w)
                
                if len(new_entity) >0: # last entity
                    arg_span_original_text, arg_span = find_arg_span_original_text(new_entity, context_words, trigger_start, trigger_end, 
                    head_only=args.head_only, doc=doc)
                    if arg_span: 
                        predicted_set.add((arg_span[0], arg_span[1], evt_type, argname, score))
                        predicted_set_original_text.add((arg_span_original_text[0], arg_span_original_text[1], evt_type, argname, score))
        
        arguments = []
        for start, end, _, argname, score in predicted_set_original_text:
            # print(context_words[start:end + 1], argname)
            if score > args.score_th:
                arguments.append({
                    "start":start,
                    "end": end + 1,
                    "role": argname,
                    "text": " ".join(context_words[start:end + 1]),
                    "score": score
                })
        docid2doc[doc_id]["event_mentions"].append({
            "event_type": evt_type,
            "trigger": ex['event']['trigger'],
            "arguments": arguments
        })
        # assert 1==0
                    
        # get gold spans 
        # (span[0], span[1], evt_type, entity_id, argname)        
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
            # clean up span by removing `a` `the`
            if args.head_only and span[0]!=span[1]:
                span = find_head(span[0], span[1], doc=doc) 
            
            gold_set.add((span[0], span[1], evt_type, entity_id, argname))
            if args.coref:
                if span in coref_mapping[doc_id]:
                    canonical_id = coref_mapping[doc_id][span]
                    gold_canonical_set.add((canonical_id, evt_type, argname))
        
        
        pred_arg_num += len(predicted_set)
        gold_arg_num += len(gold_set)
        # check matches 
        arg_idn_num_tmp = 0
        arg_class_num_tmp =0 

        arg_idn_coref_num_tmp =0
        arg_class_coref_num_tmp =0
        for pred_arg in predicted_set:
            arg_start, arg_end, event_type, role, score = pred_arg
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
                    ic_cc.append(score)
                else:
                    ic_cf.append(score)
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
                            ic_cc.append(score)
                        else:
                            ic_cf.append(score) 
                    else:
                        if_.append(score)

        # print(predicted_set)
        # print(gold_set)
        # print(f'''arg_idn_num_tmp = {arg_idn_num_tmp},
        # arg_class_num_tmp = {arg_class_num_tmp},
        # arg_idn_coref_num_tmp = {arg_idn_coref_num_tmp},
        # arg_class_coref_num_tmp = {arg_class_coref_num_tmp}''')
        examples[key]["arg_class_coref_num_tmp"] = arg_class_coref_num_tmp
        examples[key]["exact_match"] = False
        if len(gold_set) == arg_class_num_tmp + arg_class_coref_num_tmp:
            event_idn_num += 1
            event_class_num += 1
            examples[key]["exact_match"] = True

        doc_events[doc_id].append(event_arg)

    
    pair_arg = []
    for doc_id in doc_events:
        events = doc_events[doc_id]
        for ei in range(len(events) - 1):
            for ej in range(ei + 1, len(events)):
                for entity_id in events[ei]:
                    if entity_id in events[ej]:
                        pair_arg.append((events[ei][entity_id], events[ej][entity_id]))
    # for item in pair_arg:
    #     print(item)
    pair_arg_inconsistent = [1 if x[0]['predict'] != x[1]['predict'] else 0 for x in pair_arg]
    if len(pair_arg):
        print(f'paired args: {len(pair_arg)}, inconsistent args: {sum(pair_arg_inconsistent)}, raito: {sum(pair_arg_inconsistent)/len(pair_arg)}')      
        
    if args.head_only:
        print('Evaluation by matching head words only....')
    
    
    role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_class_num)

    print("Event-level identification: P: {:.2f}".format(event_idn_num/len(examples)*100.0))
    print("Event-level : P: {:.2f}".format(event_class_num/len(examples)*100.0))

    print(f"gold arg num: {gold_arg_num}")    
    print('Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    print('Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_prec * 100.0, role_rec * 100.0, role_f * 100.0))

    if args.coref:
        role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num + arg_idn_coref_num)
        role_prec, role_rec, role_f = compute_f1(
            pred_arg_num, gold_arg_num, arg_class_num + arg_class_coref_num)

        
        print('Coref Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
            role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
        print('Coref Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
            role_prec * 100.0, role_rec * 100.0, role_f * 100.0))

        print('len:',len(ic_cc), len(ic_cf), len(if_))
        print('mean:',np.mean(ic_cc), np.mean(ic_cf), np.mean(if_))
        print('var:',np.var(ic_cc), np.var(ic_cf), np.var(if_))
        print('min:', min(ic_cc), min(ic_cf), min(if_))
        print('max:', max(ic_cc), max(ic_cf), max(if_))

    with open(args.gen_file.replace("predictions","eval_results")[:-1],'w') as f:
        f.write(json.dumps(examples,indent=True))   

    with open(args.gen_file.replace("predictions","results_for_predict")[:-1],'w') as f:
        for docid in docid2doc:
            f.write(json.dumps(docid2doc[docid]) + '\n')       



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-file',type=str,default=f'checkpoints/baseline_12/epoch_5_test_step_0_predictions.jsonl' )
    parser.add_argument('--test-file', type=str,default='data/wikievents/test_no_ontology.jsonl')
    parser.add_argument('--coref-file', type=str,default='data/wikievents/coref/test.jsonlines')
    parser.add_argument('--head-only', action='store_true',default=True)
    parser.add_argument('--coref', action='store_true',default=True)
    parser.add_argument('--dataset',type=str, default='KAIROS', choices=['ACE', 'KAIROS','AIDA'])
    parser.add_argument('--score_th', type=float,default='0.0')
    arg_scorer = parser.parse_args() 

    scorer(arg_scorer)


    

                    




