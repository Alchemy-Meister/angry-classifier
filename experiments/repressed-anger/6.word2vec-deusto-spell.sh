#!/bin/sh

python ../../word2vec/dataset_w2v_embeddings.py --validation \
    --split_ratio=0.73,0.18,0.09 --delete-hashtags \
    --spell_check \
    --anger_dataset=../../datasets/split/binary_anger_dataset.csv \
    --irony_dataset=../../datasets/split/binary_irony_dataset.csv \
    --repressed_dataset=../../datasets/merged/repressed_anger_prediction_dataset_hit_processed.csv
