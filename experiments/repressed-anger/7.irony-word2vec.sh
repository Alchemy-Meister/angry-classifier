#!/bin/sh

python ../../word2vec/dataset_w2v.py --validation \
    --split_ratio=0.73,0.18,0.09 \
    ../../datasets/split/binary_irony_dataset.csv