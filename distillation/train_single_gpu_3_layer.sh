#!/bin/bash

python3 train.py \
    --student_type distilkobert \
    --student_config training_configs/distilkobert_3_layer.json \
    --teacher_type kobert \
    --teacher_name monologg/kobert-lm \
    --alpha_ce 0.33 --alpha_mlm 0.33 --alpha_cos 0.33 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path serialization_dir/3_layer \
    --data_file data/dump.kobert.pickle \
    --token_counts data/token_counts.kobert.pickle \
    --force \
    --batch_size 16 \
    --gradient_accumulation_steps 8 \
    --student_pretrained_weights serialization_dir/3_layer.pth
