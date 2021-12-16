import os 
import argparse 
import torch 
import logging 
import json 


import pytorch_lightning as pl 
from transformers import BartTokenizer, BartConfig
from transformers.models.bart.modeling_bart import BartEncoder
from transformers import AdamW, get_linear_schedule_with_warmup

from .network import BartGen
from .constrained_gen import BartConstrainedGen
import torch.nn as nn

logger = logging.getLogger(__name__)

class GenIEFinetuneModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams.update(vars(args))
    

        self.config=BartConfig.from_pretrained('facebook/bart-large')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>',' <tgr>',' <tag>'])

        self.model = BartConstrainedGen(self.config,self.tokenizer)
        self.model.resize_token_embeddings()

        self.loss = nn.MSELoss()

    # def forward(self, inputs):
    
    #     return self.model(**inputs)


    def training_step(self, batch, batch_idx):
        '''
        processed_ex = {
                            'event_idx',
                            'doc_key': ex['doc_key'],
                            'input_token_ids_1',
                            'input_attn_mask_1',
                            'argument_mask_1',
                            'input_token_ids_2',
                            'input_attn_mask_2',
                            'argument_mask_2',
                        }
        '''
        inputs_1 = {
                    "input_ids": batch["input_token_ids_1"],
                    "attention_mask": batch["input_attn_mask_1"]
                }
        encoder_outputs_1 = self.model.transformer.encoder(**inputs_1)
        inputs_2 = {
                    "input_ids": batch["input_token_ids_2"],
                    "attention_mask": batch["input_attn_mask_2"]
                }
        encoder_outputs_2 = self.model.transformer.encoder(**inputs_2)

        argument_hidden_state_1 = encoder_outputs_1.last_hidden_state[batch['argument_mask_1']]
        argument_hidden_state_2 = encoder_outputs_2.last_hidden_state[batch['argument_mask_2']]

        loss = self.loss(argument_hidden_state_1, argument_hidden_state_2)

        log = {
            'train/loss': loss, 
        } 
        return {
            'loss': loss, 
            'log': log 
        }
    

    def validation_step(self,batch, batch_idx):
        inputs_1 = {
                    "input_ids": batch["input_token_ids_1"],
                    "attention_mask": batch["input_attn_mask_1"]
                }
        encoder_outputs_1 = self.model.transformer.encoder(**inputs_1)
        inputs_2 = {
                    "input_ids": batch["input_token_ids_2"],
                    "attention_mask": batch["input_attn_mask_2"]
                }
        encoder_outputs_2 = self.model.transformer.encoder(**inputs_2)

        argument_hidden_state_1 = encoder_outputs_1.last_hidden_state[batch['argument_mask_1']]
        argument_hidden_state_2 = encoder_outputs_2.last_hidden_state[batch['argument_mask_2']]

        loss = self.loss(argument_hidden_state_1, argument_hidden_state_2)
        return loss
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))
        log = {
            'val/loss': avg_loss, 
        } 
        return {
            'loss': avg_loss, 
            'log': log 
        }
        
        
        

    def test_step(self, batch, batch_idx):

        raise ValueError("no test")

    def test_epoch_end(self, outputs):
        raise ValueError("no test")


    def configure_optimizers(self):
        self.train_len = len(self.train_dataloader())
        if self.hparams.max_steps > 0:
            t_total = self.hparams.max_steps
            self.hparams.num_train_epochs = self.hparams.max_steps // self.train_len // self.hparams.accumulate_grad_batches + 1
        else:
            t_total = self.train_len // self.hparams.accumulate_grad_batches * self.hparams.num_train_epochs

        logger.info('{} training steps in total.. '.format(t_total)) 
        
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        # scheduler is called only once per epoch by default 
        scheduler =  get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'linear-schedule',
        }

        return [optimizer, ], [scheduler_dict,]