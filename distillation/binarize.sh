#!/bin/bash
python3 scripts/binarized_data.py \
    --file_path data/dump.txt \
    --tokenizer_type kobert \
    --tokenizer_name kobert \
    --dump_file data/binarized_text