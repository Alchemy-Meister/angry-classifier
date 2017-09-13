#!/bin/sh

python ../../word2vec/dataset_w2v.py --validation \
    --split_ratio=0.70,0.15,0.15 --delete-hashtags \
    --spell_check \
    ../../datasets/merged/first_phase_binary_anger_dataset.csv