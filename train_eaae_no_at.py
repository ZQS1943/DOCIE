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

    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    if not os.path.exists(args.data_file):
        os.makedirs(args.data_file)

    

    config = BartConfig.from_pretrained('facebook/bart-large')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    tokenizer.add_tokens([' <arg>',' <tgr>',' <tag>', ' </tag>'])
    model = BartConstrainedGen(config, tokenizer)
    model.resize_token_embeddings()
    device = f'cuda:{args.gpus}'
    model.to(device)

    if args.load_ckpt:
        print(f"load from {args.load_ckpt}")
        model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict']) 
    # assert 1==0
    

    if args.eval_only:
        # source = './data/wikievents/test_info_no_ontology.jsonl'
        # target = f'./{args.data_file}/test_data_tag_gold_args_info.jsonl'
        source = './data/wikievents/test_no_ontology.jsonl'
        target = f'tmp.jsonl'
        get_data_tag_only(source = source, target = target, tokenizer = tokenizer)
        assert 1==0

        # eval_dataset = IEDataset('preprocessed_KAIROS/test.jsonl')   
        eval_dataset = IEDataset(target) 
        eval_dataloader = DataLoader(eval_dataset, num_workers=2, 
                collate_fn=my_collate,
                batch_size=args.eval_batch_size, 
                shuffle=False)

        pbar_et = tqdm(total=len(eval_dataloader))
        result_dir = (args.load_ckpt).replace(".ckpt","_test_predictions.jsonl")
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
        test_file = 'data/wikievents/test_info_no_ontology.jsonl'
        coref_file = 'data/wikievents/coref/test.jsonlines'
        scorer(score_args(result_dir, test_file, coref_file))

        return 0
        

    if args.use_info:
        train_dataset = IEDataset('preprocessed/preprocessed_KAIROS_info/train.jsonl', tokenizer = tokenizer)    
    else:
        train_dataset = IEDataset('preprocessed/preprocessed_KAIROS/train.jsonl', tokenizer = tokenizer)
    
    train_dataloader = DataLoader(train_dataset,
            pin_memory=True, num_workers=2,
            collate_fn=my_collate,
            batch_size=args.train_batch_size, 
            shuffle=True)
    
    if args.use_info:
        source = './data/wikievents/train_info_no_ontology.jsonl'    
    else:
        source = './data/wikievents/train_no_ontology.jsonl'

    target = f'./{args.data_file}/train_data_tag_other.jsonl'
    get_data_tag_only(source = source, target = target, tokenizer = tokenizer, trigger_dis = args.trg_dis)
    train_dataset_tag = IEDataset(target, tokenizer = tokenizer)
    train_dataloader_tag = DataLoader(train_dataset_tag, 
            collate_fn=my_collate,
            batch_size=args.eval_batch_size, 
            shuffle=False)

    if args.use_info:
        eval_dataset = IEDataset('preprocessed/preprocessed_KAIROS_info/val.jsonl', tokenizer = tokenizer)    
    else:
        eval_dataset = IEDataset('preprocessed/preprocessed_KAIROS/val.jsonl', tokenizer = tokenizer)
    eval_dataloader = DataLoader(eval_dataset, num_workers=2, 
            collate_fn=my_collate,
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

    for epoch in range(args.num_train_epochs):
        print("start training")
        pbar = tqdm(total=len(train_dataloader))
        model.train()
        for step, (batch, batch_tag) in enumerate(zip(train_dataloader, train_dataloader_tag)):
            if step and step % args.accumulate_grad_batches == 0 or step == len(train_dataloader) - 1:
                clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip_val)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad() 
            
            inputs = {
                    "input_ids": batch["input_token_ids"].to(device),
                    "attention_mask": batch["input_attn_mask"].to(device),
                    "decoder_input_ids": batch['tgt_token_ids'].to(device),
                    "decoder_attention_mask": batch["tgt_attn_mask"].to(device),   
                    "task": 0 
                }
            outputs,_ = model(**inputs)
            loss = outputs[0]
            loss1 = torch.mean(loss) 
            loss = loss1 / args.accumulate_grad_batches
            loss.backward()

            inputs = {
                    "input_ids": batch_tag["input_token_ids"].to(device),
                    "attention_mask": batch_tag["input_attn_mask"].to(device),
                    "decoder_input_ids": batch_tag['tgt_token_ids'].to(device),
                    "decoder_attention_mask": batch_tag["tgt_attn_mask"].to(device),   
                    "task": 0 
                }
            outputs,_ = model(**inputs)
            loss = outputs[0]
            loss2 = torch.mean(loss) 
            loss = loss2 / args.accumulate_grad_batches
            loss.backward()

            pbar.update(1)
            pbar.set_postfix({'loss1': float(loss1), 'loss2':float(loss2)})
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
                    "decoder_input_ids": batch['tgt_token_ids'].to(device),
                    "decoder_attention_mask": batch["tgt_attn_mask"].to(device),   
                    "task": 0 
                }
                
                outputs,_ = model(**inputs)
                loss = outputs[0]
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