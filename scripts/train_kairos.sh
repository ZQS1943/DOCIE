#!/usr/bin/env bash
set -e 
set -x 

# CKPT_NAME='gen-KAIROS-WFinetune'
CKPT_NAME='info_try'
rm -rf checkpoints/${CKPT_NAME}

# does not use informative mentions 
python train.py --model=constrained-gen --ckpt_name=${CKPT_NAME} \
    --dataset=KAIROS \
    --train_batch_size=4 \
    --learning_rate=1e-5 \
    --num_train_epochs=6 \
    --gpus 1 \
    --data_file=preprocessed_KAIROS_info_try \
    --use_info \
    --train_file=data/wikievents/train_info_no_ontology.jsonl \
    --val_file=data/wikievents/dev_info_no_ontology.jsonl \
    --test_file=data/wikievents/test_info_no_ontology.jsonl \
    # --load_ckpt=checkpoints/gen-KAIROS-finetune/default/version_59/checkpoints/epoch=0-step=42.ckpt
