#!/bin/sh

python ../../deep-learning/repressed-anger.py \
    --dataset=../../datasets/merged/repressed_anger_prediction_dataset_hit_processed.csv \
    --anger_dir=../../deep-learning/binary_anger_dataset/ \
    --anger_weights_filename=01-0.45.hdf5 \
    --anger_distribution=../../datasets/split/json/binary_anger_dataset_distribution.json \
    --irony_dir=../../deep-learning/binary_irony_dataset/ \
    --irony_weights_filename=01-0.34.hdf5 \
    --irony_distribution=../../datasets/split/json/binary_irony_dataset_distribution.json