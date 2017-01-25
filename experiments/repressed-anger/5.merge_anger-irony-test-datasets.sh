#!/bin/sh

python ../../datasets/merge_datasets.py \
    --output_filename=represed_anger_predition_dataset.csv \
    ../../datasets/split/binary_anger_dataset_test.csv \
    ../../datasets/split/binary_irony_dataset_test.csv
