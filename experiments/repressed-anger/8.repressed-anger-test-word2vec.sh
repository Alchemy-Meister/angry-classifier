#!/bin/sh

python ../../word2vec/dataset_w2v.py --max_phrase_length=31 \
    ../../datasets/merged/repressed_anger_prediction_dataset.csv