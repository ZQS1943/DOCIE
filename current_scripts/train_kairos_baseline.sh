#!/usr/bin/env bash
set -e 
set -x 

# CKPT_NAME='gen-KAIROS-WFinetune'

GPU='0'
SEED=('12' '21')
# rm -rf checkpoints/${CKPT_NAME}

# does not use informative mentions 
for zone in "${SEED[@]}"
do
CKPT_NAME='baseline_info_'${zone}

python train_baseline.py \
    --ckpt_name=${CKPT_NAME} \
    --train_batch_size=4 \
    --gpus ${GPU} \
    --learning_rate 3e-5 \
    --data_file=preprocessed/preprocessed_${CKPT_NAME} \
    --use_info \
    --seed ${zone}
    # --seed 12 \
done   
    # --seed 21
    # --load_ckpt=checkpoints/iterative_tag_other_finetune/epoch_2.ckpt \
    # --eval_only

# Event-level identification: P: 48.35
# Event-level : P: 48.35
# gold arg num: 561
# Role identification: P: 66.45, R: 54.01, F: 59.59
# Role: P: 58.33, R: 47.42, F: 52.31
# Coref Role identification: P: 67.98, R: 55.26, F: 60.96
# Coref Role: P: 59.87, R: 48.66, F: 53.69
