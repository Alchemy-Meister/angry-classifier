#!/bin/sh

python ../../deep-learning/cnn_japan/cnn.py \
    --train=../../datasets/merged/json/backup_deusto/two_phase_training/first_phase/baseline/first_phase_binary_irony_dataset_train.json \
    --validation=../../datasets/merged/json/backup_deusto/two_phase_training/first_phase/baseline/first_phase_binary_irony_dataset_validation.json \
    --test=../../datasets/merged/json/backup_deusto/two_phase_training/first_phase/baseline/first_phase_binary_irony_dataset_test.json
