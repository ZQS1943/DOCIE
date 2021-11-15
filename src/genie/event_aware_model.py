import os 
import argparse 
import torch 
import logging 
import json 


import pytorch_lightning as pl 
from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from .network import BartGen
from .constrained_gen import BartConstrainedGen
import torch.nn as nn

logger = logging.getLogger(__name__)

class GenIEEventAwareModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__() 
        self.hparams.update(vars(args))
    

        self.config=BartConfig.from_pretrained('facebook/bart-large')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.tokenizer.add_tokens([' <arg>',' <tgr>',' <tag>',' </tag>'])

        assert self.hparams.model=='constrained-gen'
        self.model = BartConstrainedGen(self.config, self.tokenizer)
        self.model.resize_token_embeddings() 

        self.loss_1 = nn.MSELoss()
        


    def forward(self, inputs):
    
        return self.model(**inputs)


    def training_step(self, batch, batch_idx):
        '''
        {
        'input_token_ids': input_token_ids,
        'input_attn_mask': input_attn_mask,
        'input_mask': input_mask,
        'compare_token_ids': compare_token_ids,
        'compare_attn_mask': compare_attn_mask,
        'compare_mask': compare_mask,
        'tgt_token_ids': tgt_token_ids,
        'tgt_attn_mask': tgt_attn_mask,
        'doc_key': doc_keys,
        'compare': compare,
        }
        '''
        inputs = {
                    "input_ids": batch["input_token_ids"],
                    "attention_mask": batch["input_attn_mask"],
                    "decoder_input_ids": batch['tgt_token_ids'],
                    "decoder_attention_mask": batch["tgt_attn_mask"],   
                    "task": 0 
                }
        outputs,encoder_last_hidden_state = self.model(**inputs)
        loss_2 = outputs[0]
        loss_2 = torch.mean(loss_2)
        log = {
            'train/loss': loss_2
        } 

        if sum(batch['compare']):
            # compare_token_ids = batch["compare_token_ids"][batch['compare']]
            # compare_attn_mask = batch["compare_attn_mask"][batch['compare']]
            decoder_input_ids = batch['tgt_token_ids'][batch['compare']]
            decoder_attention_mask = batch['tgt_attn_mask'][batch['compare']]
            # inputs = {
            #         "input_ids": compare_token_ids,
            #         "attention_mask": compare_attn_mask,
            #         "decoder_input_ids": decoder_input_ids,
            #         "decoder_attention_mask": decoder_attention_mask,   
            #         "task": 0 
            #     }
            
            inputs = {
                "input_ids": batch["compare_token_ids"],
                "attention_mask": batch["compare_attn_mask"],
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,   
                "task": 0 
            }
            outputs, encoder_last_hidden_state_compare = self.model(**inputs)
            loss_3 = outputs[0]
            loss_3 = torch.mean(loss_3)

            # batch['input_mask'][:,0] = True
            # batch['compare_mask'][:,0] = True
            
            # inputs_1 = {
            #             "input_ids": batch["input_token_ids"],
            #             "attention_mask": batch["input_attn_mask"]
            #         }
            # encoder_outputs_1 = self.model.transformer.encoder(**inputs_1)
            # inputs_2 = {
            #             "input_ids": batch["compare_token_ids"],
            #             "attention_mask": batch["compare_attn_mask"]
            #         }
            # # print(inputs_2)
            # encoder_outputs_2 = self.model.transformer.encoder(**inputs_2)
            encoder_last_hidden_state = encoder_last_hidden_state[batch['compare']]
            argument_hidden_state_1 = encoder_last_hidden_state[batch['input_mask']]
            argument_hidden_state_2 = encoder_last_hidden_state_compare[batch['compare_mask']]

            loss_1 = self.loss_1(argument_hidden_state_1, argument_hidden_state_2)

            def get_magnitude(number):
                number = float(number)
                s = str(number)
                number = s.split('.')
                if s[0] == '0':
                    if number[1].startswith('0000'):
                        return 0.00001
                    if number[1].startswith('000'):
                        return 0.0001
                    if number[1].startswith('00'):
                        return 0.001
                    if number[1].startswith('0'):
                        return 0.01
                    return 0.1
                else:
                    return int('1'+'0'*(len(number[0]) - 1))
            if self.hparams.lambda_value == -1:
                loss = loss_2 + self.hparams.lambda_value_3 * loss_3 + get_magnitude(loss_2 / loss_1) * loss_1
            else:
                loss = loss_2 + self.hparams.lambda_value_3 * loss_3 + self.hparams.lambda_value * loss_1 
                # loss = loss_2 + self.hparams.lambda_value * loss_1 
            if torch.isnan(loss):
                return None
            print('loss : ',float(loss_1), float(loss_2), float(loss_3), float(loss))
            log = {
                'train/loss': loss
            }   
            
            return {
                'loss':loss,
                'log':log
            } 
        if torch.isnan(loss_2):
            return None
        return {
            'loss': loss_2, 
            'log': log 
        }
    

    def validation_step(self,batch, batch_idx):
        inputs = {
                    "input_ids": batch["input_token_ids"],
                    "attention_mask": batch["input_attn_mask"],
                    "decoder_input_ids": batch['tgt_token_ids'],
                    "decoder_attention_mask": batch["tgt_attn_mask"],   
                    "task": 0 
                }
        outputs, _ = self.model(**inputs)
        loss_2 = outputs[0]
        loss_2 = torch.mean(loss_2)
        log = {
            'val/loss': loss_2
        } 
        return loss_2
        
    def validation_epoch_end(self, outputs):
        avg_loss = torch.mean(torch.stack(outputs))
        log = {
            'val/loss': avg_loss, 
        } 
        return {
            'val/loss': avg_loss, 
            'log': log 
        }
        
        
        

    def test_step(self, batch, batch_idx):
        if self.hparams.sample_gen:
            sample_output = self.model.generate(batch['input_token_ids'], do_sample=True, 
                                top_k=20, top_p=0.95, max_length=30, num_return_sequences=1,num_beams=1,
                            )
        else:
            sample_output = self.model.generate(batch['input_token_ids'], do_sample=False, 
                                max_length=30, num_return_sequences=1,num_beams=1,
                            )
        
        sample_output = sample_output.reshape(batch['input_token_ids'].size(0), 1, -1)
        doc_key = batch['doc_key'] # list 
        tgt_token_ids = batch['tgt_token_ids']

        return (doc_key, sample_output, tgt_token_ids) 

    def test_epoch_end(self, outputs):
        # evaluate F1 
        with open('checkpoints/{}/predictions.jsonl'.format(self.hparams.ckpt_name),'w') as writer:
            for tup in outputs:
                for idx in range(len(tup[0])):
                    
                    pred = {
                        'doc_key': tup[0][idx],
                        'predicted': self.tokenizer.decode(tup[1][idx].squeeze(0), skip_special_tokens=True),
                        'gold': self.tokenizer.decode(tup[2][idx].squeeze(0), skip_special_tokens=True) 
                    }
                    writer.write(json.dumps(pred)+'\n')

        return {} 


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