#!/bin/bash

export NODE_RANK=0
export N_NODES=1

export N_GPU_NODE=8
export WORLD_SIZE=8
export MASTER_PORT='8000'
export MASTER_ADDR='127.0.0.1'

pkill -f 'python3 -u train.py'

python3 -m torch.distributed.launch \
    --nproc_per_node=$N_GPU_NODE \
    --nnodes=$N_NODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train.py \
        --force \
        --n_gpu $WORLD_SIZE \
        --student_type distilkobert \
        --student_config training_configs/distilkobert_3_layer.json \
        --teacher_type kobert \
        --teacher_name monologg/kobert-lm \
        --alpha_ce 0.33 --alpha_mlm 0.33 --alpha_cos 0.33 --alpha_clm 0.0 --mlm \
        --freeze_pos_embs \
        --dump_path serialization_dir/3_layer \
        --data_file data/dump.kobert.pickle \
        --token_counts data/token_counts.kobert.pickle \
        --fp16 \
        --batch_size 20 \
        --student_pretrained_weights serialization_dir/3_layer.pth
