#!/bin/sh

python ../../word2vec/dataset_w2v.py --validation --spell_check \
    ../../datasets/merged/binary_anger_dataset.csv
