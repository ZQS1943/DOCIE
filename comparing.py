import os 
import json 
import argparse 
from copy import deepcopy
import spacy 
from spacy import displacy 
import re 
from collections import defaultdict

def find_head(arg_start, arg_end, doc):
    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <=arg_end:
        if doc[cur_i].head.i == cur_i:
            # self is the head 
            break 
        else:
            cur_i = doc[cur_i].head.i
        
    arg_head = cur_i
    
    return (arg_head, arg_head)

def extract_args_from_template(predicted, template, ontology_dict, evt_type):
    # extract argument text 
    template_words = template.strip().split()
    predicted_words = predicted.strip().split()    
    predicted_args = defaultdict(list) # argname -> List of text 
    t_ptr= 0
    p_ptr= 0 

    while t_ptr < len(template_words) and p_ptr < len(predicted_words):
        if re.match(r'<(arg\d+)>', template_words[t_ptr]):
            m = re.match(r'<(arg\d+)>', template_words[t_ptr])
            arg_num = m.group(1)
            arg_name = ontology_dict[evt_type][arg_num]

            if predicted_words[p_ptr] == '<arg>':
                # missing argument
                p_ptr +=1 
                t_ptr +=1  
            else:
                arg_start = p_ptr 
                while (p_ptr < len(predicted_words)) and (predicted_words[p_ptr] != template_words[t_ptr+1]):
                    p_ptr+=1 
                arg_text = predicted_words[arg_start:p_ptr]
                predicted_args[arg_name].append(arg_text)
                t_ptr+=1 
                # aligned 
        else:
            t_ptr+=1 
            p_ptr+=1 
    
    return dict(predicted_args)

def find_arg_span(arg, context_words, trigger_start, trigger_end, head_only=False, doc=None):
    match = None 
    arg_len = len(arg)
    min_dis = len(context_words) # minimum distance to trigger 
    for i, w in enumerate(context_words):
        if context_words[i:i+arg_len] == arg:
            if i < trigger_start:
                dis = abs(trigger_start-i-arg_len)
            else:
                dis = abs(i-trigger_end)
            if dis< min_dis:
                match = (i, i+arg_len-1)
                min_dis = dis 
    
    if match and head_only:
        assert(doc!=None)
        match = find_head(match[0], match[1], doc)
    return match 

def load_ontology(dataset):
        '''
        Read ontology file for event to argument mapping.
        ''' 
        ontology_dict ={} 
        with open('./data/event_role_{}.json'.format(dataset),'r') as f:
            ontology_dict = json.load(f)

        for evt_name, evt_dict in ontology_dict.items():
            for i, argname in enumerate(evt_dict['roles']):
                evt_dict['arg{}'.format(i+1)] = argname
                # argname -> role is not a one-to-one mapping 
                if argname in evt_dict:
                    evt_dict[argname].append('arg{}'.format(i+1))
                else:
                    evt_dict[argname] = ['arg{}'.format(i+1)]
        
        return ontology_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--baseline-file',type=str, default='checkpoints/baseline/predictions.jsonl')
    parser.add_argument('--eaae_no_iter-file',type=str, default='checkpoints/comparing_40_0.5_42/epoch_3_test_step_0_predictions.jsonl')
    parser.add_argument('--eaae-file',type=str, default='checkpoints/comparing_40_0.5_42/epoch_3_test_step_2_predictions.jsonl')
    # parser.add_argument('--result-file',type=str, default='checkpoints/gen-KAIROS-m-s-pred/predictions.jsonl')
    # parser.add_argument('--result-file-better',type=str, default='chechpoints/gen-KAIROS-m-pred/predictions.jsonl')
    parser.add_argument('--test-file', type=str, default='data/wikievents/test_no_ontology.jsonl')
    parser.add_argument('--gold', action='store_true')
    args = parser.parse_args() 

    ontology_dict = load_ontology('KAIROS')

    render_dicts = [] 

    pred_lines = open(args.result_file, 'r').read().splitlines()
    pred_lines_better = open(args.result_file_better, 'r').read().splitlines()
    ptr_pred = 0
    with open(args.test_file,'r') as f:
        for line in f:
            doc = json.loads(line)
            # use sent_id for ACE 
            context_words = doc['tokens']
            render_dict = {
                "text":' '.join(context_words),
                "ents": [],
                "title": '{}_compare'.format(doc['doc_id']),
                "bg": [],
            }
            word2char = {} # word index to start, end char index (end is not inclusive)
            ptr =0  
            for idx, w in enumerate(context_words):
                word2char[idx] = (ptr, ptr+ len(w))
                ptr = word2char[idx][1] +1  
            
            links = [] # (start_word, end_word, label)
            links_gold = []
            link_dict = {} # {"start_end":[labels]}

            
            for eidx, e in enumerate(doc['event_mentions']):
                evt_type = e['event_type']
                if evt_type not in ontology_dict:
                    continue
                label = 'E{}-{}'.format(eidx, e['event_type']) 
                predicted = json.loads(pred_lines[ptr_pred])
                predicted_better = json.loads(pred_lines_better[ptr_pred])
                ptr_pred += 1
                print(ptr_pred,eidx, doc["doc_id"], predicted["doc_key"], len(doc['event_mentions']))
                print(predicted)
                print(predicted_better)
                assert doc["doc_id"] == predicted["doc_key"] and doc["doc_id"] == predicted_better["doc_key"]
                filled_template = predicted['predicted']
                filled_template_better = predicted_better['predicted']

                
                trigger_start= e['trigger']['start']
                trigger_end = e['trigger']['end'] -1 
                trigger_tup = [trigger_start, trigger_end, label]
                links.append(trigger_tup)
                links_gold.append(trigger_tup)
                if f"{trigger_start}_{trigger_end}" not in link_dict:
                    link_dict[f"{trigger_start}_{trigger_end}"] = []
                link_dict[f"{trigger_start}_{trigger_end}"].append(label)

                # use gold arguments 
                for arg in e['arguments']:
                    label = 'E{}-{}'.format(eidx, arg['role']) 
                    ent_id = arg['entity_id']
                    # get entity span 
                    matched_ent = [entity for entity in doc['entity_mentions'] if entity['id'] == ent_id][0]
                    arg_start = matched_ent['start']
                    arg_end = matched_ent['end'] -1 
                    links_gold.append([arg_start, arg_end, label])
                    if f"{arg_start}_{arg_end}" not in link_dict:
                        link_dict[f"{arg_start}_{arg_end}"] = []
                    link_dict[f"{arg_start}_{arg_end}"].append(label)

                # use predicted arguments 
                template = ontology_dict[evt_type]['template']
                # extract argument text 
                predicted_args = extract_args_from_template(filled_template,template, ontology_dict, evt_type)
                predicted_args_better = extract_args_from_template(filled_template_better,template, ontology_dict, evt_type)
                # get trigger 
                # extract argument span
                for argname in predicted_args:
                    for argtext in predicted_args[argname]:
                        arg_span = find_arg_span(argtext, context_words, 
                            trigger_start, trigger_end, head_only=False, doc=None) 
                        if arg_span:# if None means hullucination
                            label = 'E{}-{}'.format(eidx, argname) 
                            links.append([arg_span[0], arg_span[1], label])
                            if f"{arg_span[0]}_{arg_span[1]}" not in link_dict:
                                link_dict[f"{arg_span[0]}_{arg_span[1]}"] = []
                            link_dict[f"{arg_span[0]}_{arg_span[1]}"].append(label)
                            
            
            # for span in link_dict:
            #     if len(link_dict[span]) > 1:
            #         roles = set(x.split('-')[-1] for x in link_dict[span])
            #         print(span, roles)
            #         if len(roles) > 1:
            #             start, end = span.split("_")
            #             links.append((int(start),int(end),"multi_roles"))
            sorted_links = sorted(links, key=lambda x: x[0]) # sort by start idx 
            sorted_links_gold = sorted(links_gold, key=lambda x: x[0]) # sort by start idx 
            # print(sorted_links)
            # print(sorted_links_gold)
            for idx, span in enumerate(sorted_links):
                if span not in sorted_links_gold:
                    # print(sorted_links[idx][-1])
                    sorted_links[idx][-1] = sorted_links[idx][-1] + "_in_pred"
            # print(sorted_links)
            for span in sorted_links_gold:
                if span not in sorted_links:
                    span[-1] = span[-1] + "_in_gold"
                    sorted_links.append(span)
                    # print(sorted_links)
                    # assert 1==0
            sorted_links = sorted(sorted_links, key=lambda x: x[0]) # sort by start idx 
            # print(sorted_links)
            # print(sorted_links_gold)
            # assert 1==0
            # print(sorted_links)
            # print(link_dict)
                    
            # assert 1==0
                
            for tup in sorted_links:
                arg_start, arg_end,  arg_name = tup 
                label = arg_name 
                render_dict["ents"].append({
                    "start": word2char[arg_start][0],
                    "end": word2char[arg_end][1],
                    "label": label, 
                })
                render_dict["bg"].append("rgb(163, 59, 59)")
            render_dicts.append(render_dict)


        

    file_name = args.result_file.split('.')[0] + "_comparing"
    # if args.gold:
    #     file_name += '.gold'

    html = displacy.render(render_dicts, style="ent", manual=True, page=True)

    with open('{}.html'.format(file_name), 'w') as f:
        f.write(html)

