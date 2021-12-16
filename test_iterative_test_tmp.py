import argparse 
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

    print(f"load from {args.load_ckpt}")
    model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict']) 
    
    if args.use_info:
        eval_dataset = IEDataset('preprocessed_KAIROS_info/test.jsonl')
    else:
        eval_dataset = IEDataset('preprocessed_KAIROS/test.jsonl')
    
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
                input = batch['input_token_ids'].to(device)
                sample_output = model.generate(input, do_sample=False, max_length=30, num_return_sequences=1,num_beams=1,)

                sample_output = sample_output.reshape(batch['input_token_ids'].size(0), 1, -1)
                doc_key = batch['doc_key'] # list 
                tgt_token_ids = batch['tgt_token_ids']
                
                for idx in range(len(doc_key)):
                    pred = {
                        'doc_key': doc_key[idx],
                        'predicted': tokenizer.decode(sample_output[idx].squeeze(0), skip_special_tokens=True),
                        'gold': tokenizer.decode(tgt_token_ids[idx].squeeze(0), skip_special_tokens=True) 
                    }
                    writer.write(json.dumps(pred)+'\n')
                pbar_et.update(1)
        
        print("start scoring")
        if args.use_info:
            test_file = 'data/wikievents/test_info_no_ontology.jsonl'
        else:
            test_file = 'data/wikievents/test_no_ontology.jsonl'
        coref_file = 'data/wikievents/coref/test.jsonlines'
        # test_file = f'data/wikievents/10fold/fold_{args.fold_num}/test.jsonl'
        # coref_file = f'data/wikievents/10fold/fold_{args.fold_num}/test_coref.jsonl'
        scorer(score_args(result_dir, test_file, coref_file))

        if iter_step == args.num_iterative_epochs - 1:
            print('-'*80)
            continue
        data_dir = result_dir.replace("predictions","results_for_predict")[:-1]
        target = f'{args.data_file}/test_step_{iter_step}_data.json'
        print("start getting new testset")
        get_data_tag_only(source = data_dir, target = target, tokenizer = tokenizer)

        eval_dataset = IEDataset(target)
        eval_dataloader = DataLoader(eval_dataset, 
                    collate_fn=my_collate,
                    batch_size=args.eval_batch_size, 
                    shuffle=False)


if __name__ == "__main__":
    main()