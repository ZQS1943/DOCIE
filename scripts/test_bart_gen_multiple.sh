#!/usr/bin/env bash
set -e 
set -x 


GPU='0'
TRG_DIS='40'
DATASET='KAIROS'
CKPTS=('comparing_40_0.5_21' 'comparing_0.5_40_42' 'comparing_40_0.4' 'comparing_40_0.3' 'comparing_0.5_40_21' 'comparing_40_0.5_42' 'comparing_40_0.1_42' 'comparing_40_0.5_9' 'comparing_40_1' 'comparing_40_0.7' 'comparing_40_0.5_25' 'comparing_40_10' 'comparing_40_0.5_250' 'comparing_0.5_40_12' 'comparing_40_1_25' 'comparing_0.540_42' 'comparing_0.5_40_9' 'comparing_40_0.5' 'comparing_40_0.1_9' 'comparing_40_0.5_12' 'comparing_40_0.5_100' 'comparing_40_1_9' 'comparing_40_0.1' 'comparing_40_1_42' 'comparing_40_0.6')
EPOCH='4'

 
for ckpt in "${CKPTS[@]}"
do
python src/test_bart_gen.py \
     --load_ckpt=checkpoints/${CKPT_NAME}/epoch_${EPOCH}.ckpt \
     --num_iterative_epochs 1 \
     --gpus ${GPU} \
     --trg_dis ${TRG_DIS} \
     --dataset ${DATASET} \
     --data_file 'preprocessed/'${CKPT_NAME}
done