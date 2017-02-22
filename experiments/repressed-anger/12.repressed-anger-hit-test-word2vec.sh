#!/bin/sh

python ../../word2vec/dataset_w2v.py --max_phrase_length=29 \
    --delete-hashtags \
    ../../datasets/merged/repressed_anger_prediction_dataset_hit_processed.csv