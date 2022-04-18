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


from src.genie.scorer_class import scorer_bert_crf
from src.model.constrained_gen import BartConstrainedGen
from src.data.get_data import get_data_seq


logger = logging.getLogger(__name__)

import os
from args.options import parse_arguments
from transformers import set_seed, AdamW, get_linear_schedule_with_warmup

from transformers import BertTokenizer, BertForTokenClassification, BertModel,BertPreTrainedModel, BertConfig
from src.data.data import IEDataset, my_collate_comparing, my_collate_seq

from tqdm import tqdm
import json
from torchcrf import CRF
import numpy as np

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

    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}')
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    if not os.path.exists(args.data_file):
        os.makedirs(args.data_file)

    if args.dataset == 'ACE':
        num_labels = 25
    else:
        num_labels = 85

    config = BertConfig.from_pretrained('bert-large-cased', num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    model = BERT_CRF.from_pretrained('bert-large-cased', config = config)
    device = f'cuda:{args.gpus}'
    model.to(device)

    print(f"load from {args.load_ckpt}")
    model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict']) 
    

    if args.dataset == "ACE":
        source = './data/ace05/test.wikievents.coref.json'
    elif args.dataset == "KAIROS":
        if args.use_info:
            source = './data/wikievents/test_info_no_ontology.jsonl'
        else:
            source = './data/wikievents/test_no_ontology.jsonl'
    target = f'./{args.data_file}/test_data.jsonl'
    label_to_role = get_data_seq(source = source, target = target, tokenizer = tokenizer, dataset = args.dataset)
    print(label_to_role)
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
            predicted_labels = model(**inputs)

            
            for idx in range(len(batch["input_token_ids"])):
                predicted_label = predicted_labels[idx]
                tokens = tokenizer.convert_ids_to_tokens(batch["input_token_ids"][idx])
                

                predicted_arguments = []
                if sum(predicted_label)!=0:
                    is_zero = np.array(predicted_label) == 0
                    
                    start = 0
                    end = 0
                    while False in is_zero[end:]:
                        start = list(is_zero).index(False, end)

                        for end in range(start + 1,len(predicted_label)):
                            if predicted_label[end] != predicted_label[start]:
                                break

                        predicted_arguments.append((tokens[start: end], label_to_role[int(predicted_label[start])]))

                gold_arguments = []
                if sum(batch["labels"][idx]) !=0:
                    gold_output = batch["labels"][idx]
                    is_zero = gold_output == 0

                    start = 0
                    end = 0
                    while False in is_zero[end:]:
                        start = list(is_zero).index(False, end)

                        for end in range(start + 1,len(gold_output)):
                            if gold_output[end] != gold_output[start]:
                                break
                        
                        gold_arguments.append((tokens[start: end], label_to_role[int(gold_output[start])]))

                pred = {
                    'doc_key':batch['doc_key'][idx],
                    'predicted': predicted_arguments,
                    'gold': gold_arguments,
                }           

                
                writer.write(json.dumps(pred)+'\n')

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
    
    scorer_bert_crf(score_args(result_dir, test_file, coref_file, args.score_th, args.dataset))




if __name__ == "__main__":
    main()