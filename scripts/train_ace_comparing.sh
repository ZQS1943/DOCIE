#!/usr/bin/env bash
set -e 
set -x 

# CKPT_NAME='gen-KAIROS-WFinetune'

# SEED=('12' '21' '100')
GPU='0'
TRG_DIS='40'



SEED=('42' '9' '12' '21')
alpha='0.5'
# rm -rf checkpoints/${CKPT_NAME}

# does not use informative mentions 
for zone in "${SEED[@]}"
do
CKPT_NAME='comparing_ace_'${zone}
python train_comparing_dataset.py \
    --ckpt_name=${CKPT_NAME} \
    --train_batch_size=2 \
    --gpus ${GPU} \
    --learning_rate 3e-5 \
    --data_file=preprocessed/preprocessed_${CKPT_NAME} \
    --accumulate_grad_batches=4 \
    --alpha=${alpha} \
    --seed ${zone} \
    --trg_dis ${TRG_DIS} \
    --dataset "ACE"

    # --seed 21
    # --load_ckpt=checkpoints/iterative_tag_other_finetune/epoch_2.ckpt \
    # --eval_only
done

# Event-level identification: P: 48.35
# Event-level : P: 48.35
# gold arg num: 561
# Role identification: P: 66.45, R: 54.01, F: 59.59
# Role: P: 58.33, R: 47.42, F: 52.31
# Coref Role identification: P: 67.98, R: 55.26, F: 60.96
# Coref Role: P: 59.87, R: 48.66, F: 53.69
