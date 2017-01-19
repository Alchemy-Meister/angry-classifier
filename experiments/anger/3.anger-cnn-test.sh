#!/bin/sh

python ../../deep-learning/cnn.py --target=test \
    --load_model=../../deep-learning/binary_anger_dataset/model/model.json \
    --load_weights=../../deep-learning/binary_anger_dataset/model_weights/00-0.08.hdf5 \
    --test=../../datasets/merged/json/binary_anger_dataset_test.json