#!/bin/sh

python ../../deep-learning/cnn.py --target=test \
    --load_model=../../deep-learning/binary_irony_dataset/model/model.json \
    --load_weights=../../deep-learning/binary_irony_dataset/model_weights/01-0.19.hdf5 \
    --test=../../datasets/merged/json/binary_irony_dataset_test.json