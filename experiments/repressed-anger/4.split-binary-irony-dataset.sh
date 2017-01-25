#!/bin/sh

python ../../datasets/split-dataset.py \
    --output_filenames=binary_irony_dataset_train.csv,binary_irony_dataset_validation.csv,binary_irony_dataset_test.csv \
    --remove-hashtags=irony,ironic,sarcasm,sarcastic \
    --size_ratios=0.73,0.18,0.09 \
    ../../datasets/split/binary_irony_dataset.csv