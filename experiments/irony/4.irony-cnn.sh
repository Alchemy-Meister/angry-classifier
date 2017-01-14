#!/bin/sh

python ../../deep-learning/cnn.py \
    --train=../../datasets/merged/json/binary_irony_dataset_train.json \
    --validation=../../datasets/merged/json/binary_irony_dataset_validation.json \
    --test=../../datasets/merged/json/binary_irony_dataset_test.json