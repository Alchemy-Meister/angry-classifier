#!/bin/sh

python ../../word2vec/dataset_w2v.py --validation --delete-hashtags \
    --split_ratio=0.73,0.18,0.09 --max_phrase_length=29 --twitter_model \
    ../../datasets/split/binary_irony_dataset.csv