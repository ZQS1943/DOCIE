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

from transformers import BertTokenizer, BertForTokenClassification, BertModel, BertPreTrainedModel, BertConfig
from src.data.data import IEDataset, my_collate_comparing, my_collate_seq

from tqdm import tqdm
import json
import numpy as np
from torchcrf import CRF

class BERT_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        sequence_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, head_mask=None)[0]
        logits = self.classifier(sequence_output)
        attention_mask = torch.tensor(attention_mask, dtype=torch.uint8)
        if labels == None:
            tags = self.crf.decode(logits, mask = attention_mask)
            return tags
        else:
            loss = - self.crf(logits, labels, mask = attention_mask)
            return loss


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

    
    config = BertConfig.from_pretrained('bert-large-cased', num_labels=args.role_num)
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = BERT_CRF.from_pretrained('bert-large-cased', config = config)
    device = f'cuda:{args.gpus}'
    model.to(device)

    if args.load_ckpt:
        print(f"load from {args.load_ckpt}")
        model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict']) 
    # assert 1==0
    

    if args.dataset == "ACE":
        source = './data/ace05/train.wikievents.json'
    elif args.dataset == "KAIROS":
        if args.use_info:
            source = './data/wikievents/train_info_no_ontology.jsonl'
        else:
            source = './data/wikievents/train_no_ontology.jsonl'
    target = f'./{args.data_file}/train_data.jsonl'
    get_data_seq(source = source, target = target, tokenizer = tokenizer, dataset = args.dataset)
    train_dataset = IEDataset(target)
    train_dataloader = DataLoader(train_dataset, 
            collate_fn=my_collate_seq,
            batch_size=args.train_batch_size, 
            shuffle=True)


    if args.dataset == "ACE":
        source = './data/ace05/dev.wikievents.json'
    elif args.dataset == "KAIROS":
        if args.use_info:
            source = './data/wikievents/dev_info_no_ontology.jsonl'
        else:
            source = './data/wikievents/dev_no_ontology.jsonl'
    target = f'./{args.data_file}/dev_data.jsonl'
    get_data_seq(source = source, target = target, tokenizer = tokenizer, dataset = args.dataset)
    eval_dataset = IEDataset(target)    
    eval_dataloader = DataLoader(eval_dataset, num_workers=2, 
            collate_fn=my_collate_seq,
            batch_size=args.eval_batch_size, 
            shuffle=True)

    

    train_len = len(train_dataloader)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // train_len // args.accumulate_grad_batches + 1
    else:
        t_total = train_len // args.accumulate_grad_batches * args.num_train_epochs

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)


    
    

    min_eval_loss = 1000
    mseloss = MSELoss()
    for epoch in range(args.num_train_epochs):
        print("start training")
        pbar = tqdm(total=len(train_dataloader))
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step and step % args.accumulate_grad_batches == 0 or step == len(train_dataloader) - 1:
                clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_val)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
            
            inputs = {
                    "input_ids": batch["input_token_ids"].to(device),
                    "attention_mask": batch["input_attn_mask"].to(device),
                    "labels":batch['labels'].to(device)
                }
            loss = model(**inputs)
            loss = loss / args.accumulate_grad_batches
            loss.backward()

            pbar.update(1)
            pbar.set_postfix({'loss': float(loss)})
            # pbar.set_postfix({'loss1': float(loss1)})


            
        
        print("start evaluating on evalset")
        model.eval()
        avg_loss = []
        pbar = tqdm(total=len(eval_dataloader))
        with torch.no_grad():
            for step, batch in tqdm(enumerate(eval_dataloader)):
                inputs = {
                    "input_ids": batch["input_token_ids"].to(device),
                    "attention_mask": batch["input_attn_mask"].to(device),
                    "labels":batch['labels'].to(device)
                }
                
                loss = model(**inputs)
                loss = torch.mean(loss)
                avg_loss.append(loss)
                pbar.update(1)
        avg_loss = sum(avg_loss) / len(avg_loss)

        if avg_loss < min_eval_loss:
            min_eval_loss = avg_loss
            print(f"new better ckpt {epoch}")
        save_dir = f'./checkpoints/{args.ckpt_name}/epoch_{epoch}.ckpt'

        # model.save_pretrained(save_dir)
        torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        }, save_dir)


if __name__ == "__main__":
    main()