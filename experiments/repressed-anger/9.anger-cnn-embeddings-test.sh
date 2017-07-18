#!/bin/sh

python ../../deep-learning/cnn_deusto/cnn_with_embeddings.py --target=test \
    --load_model=../../deep-learning/binary_anger_dataset/model/model.json \
    --load_weights=../../deep-learning/binary_anger_dataset/model_weights/02-0.47-300.hdf5 \
    --test=../../datasets/split/json/binary_anger_dataset_test.json
