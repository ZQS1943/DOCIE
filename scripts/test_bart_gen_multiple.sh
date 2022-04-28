#!/usr/bin/env bash
set -e 
set -x 


GPU='0'
TRG_DIS='40'
DATASET='KAIROS'

SEED=('3' '42' '95')
 
for s in "${SEED[@]}"
do
CKPT='eaae_no_at_40_KAIROS_'${s}
python src/test_bart_gen.py \
     --load_ckpt=checkpoints/${CKPT}/epoch_3.ckpt \
     --num_iterative_epochs 3 \
     --gpus ${GPU} \
     --trg_dis ${TRG_DIS} \
     --dataset ${DATASET} \
     --data_file 'preprocessed/'${ckpt} \
     --seed ${s}
done