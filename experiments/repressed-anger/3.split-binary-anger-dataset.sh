#!/bin/sh

python ../../datasets/split-dataset.py \
    --output_filenames=binary_anger_dataset_train.csv,binary_anger_dataset_validation.csv,binary_anger_dataset_test.csv \
    --remove-hashtags=anger,angry,angrryyy,annoyed,annoying,irritated,irritating,disgasting,disgasted \
    --size_ratios=0.73,0.18,0.09 \
    ../../datasets/split/binary_anger_dataset.csv