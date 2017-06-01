#!/bin/sh

python ../../deep-learning/cnn_japan/cnn.py \
    --train=../../datasets/split/json/binary_irony_dataset_train.json \
    --validation=../../datasets/split/json/binary_irony_dataset_validation.json \
    --test=../../datasets/split/json/binary_irony_dataset_test.json
