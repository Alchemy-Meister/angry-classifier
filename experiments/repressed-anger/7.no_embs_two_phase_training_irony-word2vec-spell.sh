#!/bin/sh

python ../../word2vec/dataset_w2v.py --validation --delete-hashtags \
    --split_ratio=0.70,0.15,0.15 --max_phrase_length=29 \
    --spell_check \
    ../../datasets/merged/first_phase_binary_irony_dataset.csv