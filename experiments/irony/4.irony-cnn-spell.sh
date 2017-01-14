#!/bin/sh

python ../../deep-learning/cnn.py \
    --train=../../datasets/merged/json/binary_irony_dataset_spell_train.json \
    --validation=../../datasets/merged/json/binary_irony_dataset_spell_validation.json \
    --test=../../datasets/merged/json/binary_irony_dataset_spell_test.json