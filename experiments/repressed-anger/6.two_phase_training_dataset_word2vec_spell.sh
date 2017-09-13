#!/bin/sh

python ../../word2vec/dataset_w2v_embeddings.py --validation \
    --split_ratio=0.70,0.15,0.15 --delete-hashtags \
    --spell_check \
    --anger_dataset=../../datasets/merged/first_phase_binary_anger_dataset.csv \
    --irony_dataset=../../datasets/merged/first_phase_binary_irony_dataset.csv \
    --repressed_dataset=../../datasets/merged/repressed_anger_prediction_dataset_hit_reprocessed_trimmed.csv \
    --serialize_csv
