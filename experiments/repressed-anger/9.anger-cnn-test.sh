#!/bin/sh

python ../../deep-learning/cnn_japan/cnn.py --target=test \
    --load_model=../../deep-learning/binary_anger_dataset/model/model.json \
    --load_weights=../../deep-learning/binary_anger_dataset/model_weights/01-0.45.hdf5 \
    --test=../../datasets/split/json/binary_anger_dataset_test.json