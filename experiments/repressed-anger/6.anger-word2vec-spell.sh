#!/bin/sh

python ../../word2vec/dataset_w2v.py --validation \
    --split_ratio=0.73,0.18,0.09 --delete-hashtags \
    --spell_check \
    ../../datasets/split/binary_anger_dataset.csv