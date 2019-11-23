#!/bin/bash

python3 scripts/token_counts.py \
    --data_file data/binarized_text.kobert.pickle \
    --token_counts_dump data/token_counts.kobert.pickle \
    --vocab_size 8002