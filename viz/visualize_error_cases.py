import os 
import json 
import argparse 
from copy import deepcopy
import spacy 
from spacy import displacy 
import re 
from collections import defaultdict
from bs4 import BeautifulSoup

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--result-file',type=str, default='checkpoints/comparing_40_0.5_9/epoch_5_test_step_4_error_cases.json')
    args = parser.parse_args() 

    render_dicts = [] 

    with open(args.result_file, 'r') as f:
        error_cases = json.loads(f.read())
    
    for case in error_cases:
        context_words = case['input_tokens']
        render_dict = {
                "text":' '.join(context_words) + "\n\nPredicted template:" + case['predicted'],
                "ents": [],
                "title": f"Event ID: {case['event_id']}",
                "bg": [],
            }

        word2char = {} # word index to start, end char index (end is not inclusive)
        ptr =0  
        for idx, w in enumerate(context_words):
            word2char[idx] = (ptr, ptr+ len(w))
            ptr = word2char[idx][1] +1 

        links = [(case['trigger']['start'], case['trigger']['end'] - 1, case['event_type'])] # (start_word, end_word, label)

        for arg in case['arguments']:
            links.append((arg['start'], arg['end'] - 1, f"{arg['role']}_{arg['type']}"))
        sorted_links = sorted(links, key=lambda x: x[0])

        for tup in sorted_links:
            arg_start, arg_end, arg_name = tup 
            label = arg_name 
            render_dict["ents"].append({
                "start": word2char[arg_start][0],
                "end": word2char[arg_end][1],
                "label": label, 
            })
        render_dicts.append(render_dict)
        

        

    file_name = args.result_file.split('.')[0]
    # if args.gold:
    #     file_name += '.gold'

    html = displacy.render(render_dicts, style="ent", manual=True, page=True)
    soup = BeautifulSoup(html)
    for i,mark in enumerate(soup.find_all('mark')):
        if '_' not in mark.span.get_text():
            mark['style'] = mark['style'].replace('#ddd', 'rgb(253, 213, 111)')
        else:
            role, types = mark.span.get_text().split('_')
            if types == 'both':
                continue
            if types == 'predicted':
                mark['style'] = mark['style'].replace('#ddd', 'rgb(255, 108, 98)')
            elif types == 'gold':
                mark['style'] = mark['style'].replace('#ddd', 'rgb(255, 108, 98)')
        soup.find_all('mark')[i].replace_with(mark)

    with open('{}.html'.format(file_name), 'w') as f:
        f.write(soup.prettify())

