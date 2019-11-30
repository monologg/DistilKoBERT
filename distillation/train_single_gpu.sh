#!/bin/bash

python3 train.py \
    --student_type distilkobert \
    --student_config training_configs/distilkobert_3_layer.json \
    --teacher_type kobert \
    --teacher_name kobert \
    --alpha_ce 5.0 --alpha_mlm 2.0 --alpha_cos 1.0 --alpha_clm 0.0 --mlm \
    --freeze_pos_embs \
    --dump_path serialization_dir/my_first_training \
    --data_file data/binarized_text.kobert.pickle \
    --token_counts data/token_counts.kobert.pickle \
    --force \
    --student_pretrained_weights serialization_dir/3_layer.pth
