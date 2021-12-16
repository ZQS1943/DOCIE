#!/usr/bin/env bash
set -e 
set -x 

for fold_num in {0..9}
do
python train.py \
    --ckpt_name=fold_${fold_num}_normal \
    --train_batch_size=4 \
    --gpus 0 \
    --learning_rate 3e-5 \
    --data_file=preprocessed_fold_${fold_num}_normal \
    --accumulate_grad_batches=4 \
    --num_train_epochs=3\
    --train_file=data/wikievents/10fold/fold_${fold_num}/train.jsonl \
    --val_file=data/wikievents/10fold/fold_${fold_num}/dev.jsonl \
    --test_file=data/wikievents/10fold/fold_${fold_num}/test.jsonl
done