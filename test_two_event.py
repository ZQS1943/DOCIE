import argparse
from collections import defaultdict 
import logging 
import os 
import random 
import timeit 
from datetime import datetime 

import torch 
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers.modeling_utils import unwrap_model 

# import wandb

from src.genie.get_new_data_file import get_new_data_file
# from src.genie.scorer_class import scorer
from src.model.constrained_gen import BartConstrainedGen
from src.data.get_data import get_data_two_event


logger = logging.getLogger(__name__)

import os
from args.options import parse_arguments
from transformers import set_seed, AdamW, get_linear_schedule_with_warmup

from transformers import BartTokenizer, BartConfig
from src.data.data import IEDataset, my_collate

from tqdm import tqdm
import json




class score_args:
    def __init__(self, gen_file, test_file, coref_file) -> None:
        self.gen_file = gen_file
        self.test_file = test_file
        self.coref_file = coref_file
        self.dataset = "KAIROS"
        self.coref = True
        self.head_only = True
            
def main():
    args = parse_arguments()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info("Training/evaluation parameters %s", args)

    set_seed(args.seed)

    

    config = BartConfig.from_pretrained('facebook/bart-large')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    tokenizer.add_tokens([' <arg>',' <tgr>',' <tag>', ' </tag>'])
    model = BartConstrainedGen(config, tokenizer)
    model.resize_token_embeddings()
    device = f'cuda:{args.gpus}'
    model.to(device)

    # print(tokenizer.convert_tokens_to_ids([';']))
    # assert 1==0

    print(f"load from {args.load_ckpt}")
    model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict']) 
    
    if args.use_info:
        source = './data/wikievents/test_info_no_ontology.jsonl'    
    else:
        source = './data/wikievents/test_no_ontology.jsonl'
    target = f'./{args.data_file}/test_data_two_event.jsonl'
    get_data_two_event(source = source, target = target, tokenizer = tokenizer)
    eval_dataset = IEDataset(target, tokenizer = tokenizer)
    eval_dataloader = DataLoader(eval_dataset, 
            collate_fn=my_collate,
            batch_size=args.eval_batch_size, 
            shuffle=False)

    results = defaultdict(lambda : defaultdict(list))
    pbar_et = tqdm(total=len(eval_dataloader))
    result_dir = (args.load_ckpt).replace(".ckpt",f"_predictions.jsonl")
    model.eval()  
    output_num = {0:0,1:0,2:0}  
    gold_num = {0:0,1:0,2:0}
    type_output = [[0,0,0],[0,0,0],[0,0,0]]
    for step, batch in enumerate(eval_dataloader):
        doc_key = batch['doc_key'] # list 
        tgt_token_ids = batch['tgt_token_ids']

        input = batch['input_token_ids'].to(device) 
        # print(input)
        # sample_output, scores = model.generate(input, do_sample=False, max_length=45, num_return_sequences=1,num_beams=1,decoder_start_token_id=0, two_template=True)
        sample_output, scores = model.generate(input, do_sample=False, max_length=45, num_return_sequences=1,num_beams=1,decoder_start_token_id=0)
        # print(sample_output)
        # assert 1==0
        
        for idx in range(len(doc_key)):
            event_idx = batch['event_idx'][idx]
            event_type = batch['event_type'][idx]
            trigger = batch['trigger'][idx]
            output_ids = sample_output[idx]
            original_output = tokenizer.decode(output_ids, skip_special_tokens=True)
            original_gold = tokenizer.decode(tgt_token_ids[idx], skip_special_tokens=True)

            # print(tokenizer.decode(input[idx], skip_special_tokens=True))
            # print(original_output)
            # # print(original_output)
            # print(event_type)
            # assert 1==0
            
            output = original_output[:-1].split(';')
            gold = original_gold[:-1].split(';')
            output_num[len(output)] += 1
            gold_num[len(gold)] += 1

            pred = {
                'doc_key': doc_key[idx],
                'predicted': '',
                'gold': '',
                'original_predicted':  original_output,
                'original_gold': original_gold,
                'event_idx':event_idx
            }

            
            if len(output) == 1:
                pred['predicted'] = output[0]
                pred['gold'] = gold[0]
                results[doc_key[idx]][event_idx[0]].append(pred)     
            else:
                pred['predicted'] = output[0]
                pred['gold'] = gold[0]
                results[doc_key[idx]][event_idx[0]].append(pred)     
                
                pred['predicted'] = output[1]
                pred['gold'] = gold[1]
                results[doc_key[idx]][event_idx[1]].append(pred)
            print('-'*10)
            # print('output:', original_output)
            # print('gold:', original_gold)
            input_tokens = tokenizer.convert_ids_to_tokens(input[idx])
            # print(input_tokens)
            # print(input_tokens.count(' <tgr>'))
            # assert 1==0
            
            print(event_type, trigger,len(output), len(gold), input_tokens.count(' <tgr>'))
            type_output[len(set(event_type))][len(output)] += 1
        pbar_et.update(1)       
    print(output_num)
    print(gold_num)
    print(type_output)
    assert 1==0
    with open(result_dir, 'w') as writer:
        writer.write(json.dumps(results,indent=True))         
        
    print("start scoring")
    if args.use_info:
        test_file = 'data/wikievents/test_info_no_ontology.jsonl'
    else:
        test_file = 'data/wikievents/test_no_ontology.jsonl'
    coref_file = 'data/wikievents/coref/test.jsonlines'
    # scorer(score_args(result_dir, test_file, coref_file))


if __name__ == "__main__":
    main()