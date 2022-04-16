#!/usr/bin/env bash
set -e 
set -x 

python eval_on_dev.py \
     --num_iterative_epochs 6 \
     --gpus 2 \
     --data_file=preprocessed/preprocessed_eval_on_dev \
     --eval_batch_size 8