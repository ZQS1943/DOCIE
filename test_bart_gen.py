import argparse
from collections import defaultdict 
import logging 
import os 
import random 
import timeit 
from datetime import datetime
from spacy.util import working_dir 

import torch 
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers.modeling_utils import unwrap_model 

# import wandb


from src.genie.scorer_class import scorer
from src.model.constrained_gen import BartConstrainedGen
from src.data.get_data import get_data_tag_only, get_data_normal


logger = logging.getLogger(__name__)

import os
from args.options import parse_arguments
from transformers import set_seed, AdamW, get_linear_schedule_with_warmup

from transformers import BartTokenizer, BartConfig
from src.data.data import IEDataset, my_collate

from tqdm import tqdm
import json




class score_args:
    def __init__(self, gen_file, test_file, coref_file, score_th = 0, dataset = "KAIROS") -> None:
        self.gen_file = gen_file
        self.test_file = test_file
        self.coref_file = coref_file
        self.score_th = score_th
        self.dataset = dataset
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


    
    print(f"load from {args.load_ckpt}")
    model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict']) 
    
    if args.dataset == "ACE":
        source = './data/ace05/test.wikievents.json'
        target = f'./{args.data_file}/test_data.jsonl'
        get_data_normal(source = source, target = target, tokenizer = tokenizer, dataset = args.dataset) 
        eval_dataset = IEDataset(target)
    elif args.dataset == "KAIROS":
        if args.use_info:
            eval_dataset = IEDataset('preprocessed/preprocessed_KAIROS_info/test.jsonl', tokenizer = tokenizer)
        else:
            eval_dataset = IEDataset('preprocessed/preprocessed_KAIROS/test.jsonl', tokenizer = tokenizer)
    
    # eval_dataset = IEDataset(f'preprocessed_fold_{args.fold_num}_normal/test.jsonl')
    # data_dir = "checkpoints/iterative_fast_5e-5/epoch_2_test_step_0_results_for_predict.json"
    # target = 'tmp_0.json'
    # print("start getting new testset")
    # get_data_tag_only(source = data_dir, target = target, tokenizer = tokenizer)
    # eval_dataset = IEDataset('preprocessed_iterative_fast_5e_5_50/test_data_tag_gold_args.jsonl')
    # eval_dataset = IEDataset(target)
    eval_dataloader = DataLoader(eval_dataset, 
            collate_fn=my_collate,
            batch_size=args.eval_batch_size, 
            shuffle=False)


    for iter_step in range(args.num_iterative_epochs):
        pbar_et = tqdm(total=len(eval_dataloader))
        result_dir = (args.load_ckpt).replace(".ckpt",f"_test_step_{iter_step}_predictions.jsonl")
        model.eval()
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
                        if w != 'Ġ.' and (w[0] == 'Ġ' or w == " <arg>" or len(word_score) == 0) :
                            word_score.append([float(s)])
                        else:
                            word_score[-1].append(float(s))
                    words = list(filter(lambda x: len(x), output.split(' ')))[:len(word_score)]
                    
                    # print('*'*10)
                    # # print(tokenizer.convert_ids_to_tokens(output_ids))
                    # # print(tokens)
                    # # print(words)
                    # print(tokens)
                    # print(words)
                    # print(word_score)
                    assert len(word_score) == len(words)
                    # for _ in zip(word_score, words):
                    #     print(_)
                    # for idx
                    word_score = [sum(x)/len(x) for x in word_score]
                    

                    pred = {
                        'doc_key': doc_key[idx],
                        'predicted': output,
                        'gold': gold_output,
                        'scores': word_score
                    }
                    writer.write(json.dumps(pred)+'\n')
                
                # assert 1==0
        
                # # tokens = tokenizer.convert_ids_to_tokens(sample_output[0])
                # # print(tokens)
                # # print(scores)
                # # assert 1==0
                # sample_output = sample_output.reshape(batch['input_token_ids'].size(0), 1, -1)
                # scores = scores.reshape(batch['input_token_ids'].size(0), 1, -1)

                            
                
                # for idx in range(len(doc_key)):
                #     pred = {
                #         'doc_key': doc_key[idx],
                #         'predicted': tokenizer.decode(sample_output[idx].squeeze(0), skip_special_tokens=True),
                #         'gold': tokenizer.decode(tgt_token_ids[idx].squeeze(0), skip_special_tokens=True) ,
                #         'scores': scores[idx].squeeze(0).tolist()
                #     }
                #     writer.write(json.dumps(pred)+'\n')
                pbar_et.update(1)
        
        print("start scoring")
        if args.dataset == "ACE":
            test_file = 'data/ace05/test.wikievents.coref.json'
            coref_file = None
        elif args.dataset == "KAIROS":
            if args.use_info:
                test_file = 'data/wikievents/test_info_no_ontology.jsonl'
            else:
                test_file = 'data/wikievents/test_no_ontology.jsonl'
        
            coref_file = 'data/wikievents/coref/test.jsonlines'
        # test_file = f'data/wikievents/10fold/fold_{args.fold_num}/test.jsonl'
        # coref_file = f'data/wikievents/10fold/fold_{args.fold_num}/test_coref.jsonl'
        scorer(score_args(result_dir, test_file, coref_file, args.score_th, args.dataset))

        if iter_step == args.num_iterative_epochs - 1:
            print('-'*80)
            continue
        data_dir = result_dir.replace("predictions","results_for_predict")[:-1]
        target = f'{args.data_file}/test_step_{iter_step}_data.json'
        print("start getting new testset")
        get_data_tag_only(source = data_dir, target = target, tokenizer = tokenizer, trigger_dis = args.trg_dis, dataset = args.dataset)

        eval_dataset = IEDataset(target, tokenizer = tokenizer)
        eval_dataloader = DataLoader(eval_dataset, 
                    collate_fn=my_collate,
                    batch_size=args.eval_batch_size, 
                    shuffle=False)


if __name__ == "__main__":
    main()