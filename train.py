import argparse 
import logging 
import os 
import random 
import timeit 
from datetime import datetime 

import torch 
# import wandb 
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything



from src.data.KAIROS_data_module import KAIROSDataModule 
from src.model.model import GenIEModel 


logger = logging.getLogger(__name__)

import os
from args.options import parse_arguments
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def main():
    args = parse_arguments()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Set seed
    seed_everything(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    
    if not args.ckpt_name:
        d = datetime.now() 
        time_str = d.strftime('%m-%dT%H%M')
        args.ckpt_name = '{}_{}lr{}_{}'.format(args.model,  args.train_batch_size * args.accumulate_grad_batches, 
                args.learning_rate, time_str)


    args.ckpt_dir = os.path.join(f'./checkpoints/{args.ckpt_name}')
    
    os.makedirs(args.ckpt_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        save_top_k=args.save_top_k,
        monitor='val/loss',
        mode='min',
        save_weights_only=True,
        filename='{epoch}', # this cannot contain slashes 

    )

   


    lr_logger = LearningRateMonitor() 
    tb_logger = TensorBoardLogger('logs/')

    model = GenIEModel(args)
    dm = KAIROSDataModule(args)



    if args.max_steps < 0 :
        args.max_epochs = args.min_epochs = args.num_train_epochs 
    
    # print(args.gpus)
    # assert 1==0

    trainer = Trainer(
        logger=tb_logger,
        min_epochs=args.num_train_epochs,
        max_epochs=args.num_train_epochs, 
        gpus=args.gpus, 
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val, 
        num_sanity_val_steps=0, 
        val_check_interval=0.5, # use float to check every n epochs 
        precision=16 if args.fp16 else 32,
        callbacks = [lr_logger, checkpoint_callback],
    )  

    if args.load_ckpt:
        print(args.load_ckpt)
        if not os.path.exists(args.load_ckpt):
            print('ckpt does not exist')
            quit()
        
        model.load_state_dict(torch.load(args.load_ckpt,map_location=model.device)['state_dict']) 

    if args.eval_only: 
        dm.setup('test')
        trainer.test(model, datamodule=dm) #also loads training dataloader 
    else:
        dm.setup('fit')
        trainer.fit(model, dm) 
    

    

if __name__ == "__main__":
    main()