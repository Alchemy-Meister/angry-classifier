#!/bin/sh

python ../../deep-learning/cnn.py \
    --train=../../datasets/split/json/binary_anger_dataset_train.json \
    --validation=../../datasets/split/json/binary_anger_dataset_validation.json \
    --test=../../datasets/split/json/binary_anger_dataset_test.json