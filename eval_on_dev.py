import argparse
from collections import defaultdict
from email.policy import default 
import logging 
import os 
import random 
import timeit 
from datetime import datetime
from unittest import result
from spacy.util import working_dir 

import torch 
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers.modeling_utils import unwrap_model 

# import wandb

from src.genie.get_new_data_file import get_new_data_file
from src.genie.scorer_class import scorer
from src.model.constrained_gen import BartConstrainedGen
from src.data.get_data import get_data_tag_only


logger = logging.getLogger(__name__)

import os
from args.options import parse_arguments
from transformers import set_seed, AdamW, get_linear_schedule_with_warmup

from transformers import BartTokenizer, BartConfig
from src.data.data import IEDataset, my_collate

from tqdm import tqdm
import json




class score_args:
    def __init__(self, gen_file, test_file, coref_file, score_th = 0) -> None:
        self.gen_file = gen_file
        self.test_file = test_file
        self.coref_file = coref_file
        self.score_th = score_th
        self.dataset = "KAIROS"
        self.coref = True
        self.head_only = True
            
def main(args):
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


    print(f"load from {args.load_ckpt}")
    model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict']) 

    ckpt = (args.load_ckpt).split('/')[-1][6]
    
    # eval_dataset = IEDataset('preprocessed/preprocessed_KAIROS/val.jsonl', tokenizer = tokenizer)
    eval_dataset = IEDataset('preprocessed/preprocessed_KAIROS/test.jsonl', tokenizer = tokenizer)
    eval_dataloader = DataLoader(eval_dataset, 
            collate_fn=my_collate,
            batch_size=args.eval_batch_size, 
            shuffle=False)

    results = {}

    for iter_step in range(args.num_iterative_epochs):
        pbar_et = tqdm(total=len(eval_dataloader))
        result_dir = (args.load_ckpt).replace(".ckpt",f"_test_step_{iter_step}_predictions.jsonl")
        model.eval()
        with torch.no_grad():
            with open(result_dir, 'w') as writer: 
                for step, batch in enumerate(eval_dataloader):
                    doc_key = batch['doc_key'] # list 
                    tgt_token_ids = batch['tgt_token_ids']


                    input = batch['input_token_ids'].to(device)
                    sample_output, scores = model.generate(input, do_sample=False, max_length=30, num_return_sequences=1,num_beams=1, decoder_start_token_id=0)
                    
                    
                    for idx in range(len(doc_key)):
                        output_ids = sample_output[idx]
                        tokens = tokenizer.convert_ids_to_tokens(output_ids)[1:-1]
                        score = scores[idx][1:-1]
                        output = tokenizer.decode(output_ids, skip_special_tokens=True)
                        gold_output = tokenizer.decode(tgt_token_ids[idx], skip_special_tokens=True)

                        word_score = []
                        # print('*'*10)
                        # print(tokens)
                        # print(tokenizer.convert_ids_to_tokens(output_ids))
                        for w, s in zip(tokens, score):
                            if w[0] == 'Ġ' or w == " <arg>" or len(word_score) == 0:
                                word_score.append([float(s)])
                            else:
                                word_score[-1].append(float(s))
                        # words = list(filter(lambda x: len(x), output.split(' ')))[:len(word_score)]
                        
                        # print('*'*10)
                        # print(tokenizer.convert_ids_to_tokens(output_ids))
                        # print(tokens)
                        # print(words)
                        # assert len(word_score) == len(words)
                        # for _ in zip(word_score, words):
                        #     print(_)
                        # for idx
                        # word_score = [sum(x)/len(x) for x in word_score]
                        

                        pred = {
                            'doc_key': doc_key[idx],
                            'predicted': output,
                            'gold': gold_output,
                        }
                        writer.write(json.dumps(pred)+'\n')
                    pbar_et.update(1)
        
        print("start scoring")
        test_file = 'data/wikievents/test_no_ontology.jsonl'
        coref_file = 'data/wikievents/coref/test.jsonlines'

        results[iter_step] = scorer(score_args(result_dir, test_file, coref_file, args.score_th))

        if iter_step == args.num_iterative_epochs - 1:
            print('-'*80)
            continue
        data_dir = result_dir.replace("predictions","results_for_predict")[:-1]
        target = f'{args.data_file}/test_ckpt_{ckpt}_step_{iter_step}_data.json'
        print("start getting new testset")
        get_data_tag_only(source = data_dir, target = target, tokenizer = tokenizer, trigger_dis = args.trg_dis)

        eval_dataset = IEDataset(target, tokenizer = tokenizer)
        eval_dataloader = DataLoader(eval_dataset, 
                    collate_fn=my_collate,
                    batch_size=args.eval_batch_size, 
                    shuffle=False)

    return results


if __name__ == "__main__":
    args = parse_arguments()
    results = defaultdict(lambda : defaultdict(dict)) # para:epoch_num:iter_num

    for item in os.walk('./checkpoints/'):
        if '/comparing' in item[0] and 'ace' not in item[0] and '_9' not in item[0] and '_12' not in item[0] and '_21' not in item[0] and 'finetune' not in item[0] and '_25' not in item[0]:
            for epoch in item[2]:
                if '.ckpt' in epoch:
                    epoch_num = int(epoch[-6])
                    if epoch_num <=1:
                        continue
                    args.load_ckpt = item[0] + '/' + epoch
                    # args.load_ckpt = './checkpoints/comparing_50/epoch_4.ckpt'
                    args.load_ckpt = './checkpoints/comparing_40_0.1/epoch_5.ckpt'
                    results[item[0]][epoch_num] = 1
                    results[item[0]][epoch_num] = main(args)
                    assert 1==0
            # # args.ckpt_name = 
            # results[item[0]] = main(args)
    print(results)
    with open('./eval_on_dev_comparing.txt', 'w') as f:
        f.write(json.dumps(results))      