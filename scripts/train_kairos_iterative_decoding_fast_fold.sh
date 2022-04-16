#!/usr/bin/env bash
set -e 
set -x 

# CKPT_NAME='gen-KAIROS-WFinetune'
# rm -rf checkpoints/${CKPT_NAME}

# does not use informative mentions 
# for fold_num in {5}
# do
fold_num=6
python train_iterative_decoding_fast_fold.py \
    --ckpt_name=fold_check_${fold_num} \
    --train_batch_size=4 \
    --gpus 3 \
    --learning_rate 5e-5 \
    --data_file=preprocessed_fold_${fold_num} \
    --accumulate_grad_batches=4 \
    --fold_num ${fold_num} \
    # --load_ckpt=checkpoints/iterative_tag_other_finetune/epoch_2.ckpt \
    # --eval_only
# done
# Event-level identification: P: 48.35
# Event-level : P: 48.35
# gold arg num: 561
# Role identification: P: 66.45, R: 54.01, F: 59.59
# Role: P: 58.33, R: 47.42, F: 52.31
# Coref Role identification: P: 67.98, R: 55.26, F: 60.96
# Coref Role: P: 59.87, R: 48.66, F: 53.69