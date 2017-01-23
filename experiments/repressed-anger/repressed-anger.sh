#!/bin/sh

python ../../deep-learning/repressed-anger.py \
    --dataset=../../datasets/merged/json/binary_anger_dataset_test.json \
    --anger_dir=../../deep-learning/binary_anger_dataset/ \
    --anger_weights_filename=00-0.08.hdf5 \
    --irony_dir=../../deep-learning/binary_irony_dataset/ \
    --irony_weights_filename=01-0.19.hdf5
