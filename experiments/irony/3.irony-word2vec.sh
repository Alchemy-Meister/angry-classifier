#!/bin/sh

workon anger-detection

python ../../word2vec/dataset_w2v.py --validation \
    ../../datasets/merged/binary_irony_dataset.csv
