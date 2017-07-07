#!/bin/sh

python ../../deep-learning/cnn_deusto/cnn_with_embeddings.py \
    --embedding_weight=../../datasets/split/embedding_weights.npy \
    --train=../../datasets/split/json/binary_anger_dataset_train.json \
    --validation=../../datasets/split/json/binary_anger_dataset_validation.json \
    --test=../../datasets/split/json/binary_anger_dataset_test.json
