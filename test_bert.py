import argparse 
import logging 
import os 
import random 
import timeit 
from datetime import datetime 

import torch 
from torch import nn
from torch.nn import MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers.modeling_utils import unwrap_model 

# import wandb

from src.genie.get_new_data_file import get_new_data_file
from src.genie.scorer_class import scorer
from src.model.constrained_gen import BartConstrainedGen
from src.data.get_data import get_data_seq


logger = logging.getLogger(__name__)

import os
from args.options import parse_arguments
from transformers import set_seed, AdamW, get_linear_schedule_with_warmup

from transformers import BertTokenizer, BertForTokenClassification, BertModel
from src.data.data import IEDataset, my_collate_comparing, my_collate_seq

from tqdm import tqdm
import json

class MyBertForTokenClassification(BertForTokenClassification):

    def __init__(self, config):
        config.num_labels = 81
        super().__init__(config)



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

    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    if not os.path.exists(args.data_file):
        os.makedirs(args.data_file)

    

    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = MyBertForTokenClassification.from_pretrained('bert-large-cased')
    device = f'cuda:{args.gpus}'
    model.to(device)

    print(f"load from {args.load_ckpt}")
    model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict']) 
    # assert 1==0
    

    if args.dataset == "ACE":
        source = './data/ace05/test.wikievents.coref.json'
    elif args.dataset == "KAIROS":
        if args.use_info:
            source = './data/wikievents/test_info_no_ontology.jsonl'
        else:
            source = './data/wikievents/test_no_ontology.jsonl'
    target = f'./{args.data_file}/test_data.jsonl'
    label_to_role = get_data_seq(source = source, target = target, tokenizer = tokenizer, dataset = args.dataset)
    eval_dataset = IEDataset(target)
    eval_dataloader = DataLoader(eval_dataset, 
            collate_fn=my_collate_seq,
            batch_size=args.eval_batch_size, 
            shuffle=False)
    

    pbar_et = tqdm(total=len(eval_dataloader))
    result_dir = (args.load_ckpt).replace(".ckpt",f"_predictions.jsonl")
    model.eval()
    with open(result_dir, 'w') as writer: 
        for step, batch in enumerate(eval_dataloader):
            inputs = {
                    "input_ids": batch["input_token_ids"].to(device),
                    "attention_mask": batch["input_attn_mask"].to(device)}
            output = model(**inputs)
            logits = output.logits
            predicted_labels = torch.argmax(logits, dim=2)
            
            for idx in range(len(batch["input_token_ids"])):
                predicted_label = predicted_labels[idx]

                predicted_arguments = []
                if sum(predicted_label)!=0:
                    # print(predicted_label)
                    is_zero = predicted_label == 0
                    # print(is_zero)
                    start = 0
                    while False in is_zero[start + 1:]:
                        start = list(is_zero).index(False, start + 1)

                        for end in range(start + 1,len(predicted_label)):
                            if predicted_label[end] != predicted_label[start]:
                                break
                        
                        predicted_arguments.append((start, end, label_to_role[int(predicted_label[start])]))
                    
                    # print(predicted_arguments)
                    # assert 1==0
        

                # pred = {
                #     'doc_key': doc_key[idx],
                #     'predicted': output,
                #     'gold': gold_output,
                #     'scores': word_score
                # }
                # writer.write(json.dumps(pred)+'\n')

            pbar_et.update(1)
    
    # print("start scoring")
    # if args.use_info:
    #     test_file = 'data/wikievents/test_info_no_ontology.jsonl'
    # else:
    #     test_file = 'data/wikievents/test_no_ontology.jsonl'
    # coref_file = 'data/wikievents/coref/test.jsonlines'
    # # test_file = f'data/wikievents/10fold/fold_{args.fold_num}/test.jsonl'
    # # coref_file = f'data/wikievents/10fold/fold_{args.fold_num}/test_coref.jsonl'
    # scorer(score_args(result_dir, test_file, coref_file, args.score_th))




if __name__ == "__main__":
    main()