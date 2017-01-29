#!/bin/sh

python ../../datasets/merge_datasets.py \
    --output_filename=repressed_anger_prediction_dataset.csv \
    ../../datasets/split/binary_anger_dataset_test.csv \
    ../../datasets/split/binary_irony_dataset_test.csv
