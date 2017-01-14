#!/bin/sh

workon anger-detection

python ../../word2vec/dataset_w2v.py --validation --spell_check\
    ../../datasets/merged/binary_irony_dataset.csv
