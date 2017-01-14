#!/bin/sh

workon anger-detection

python ../../datasets/merge_datasets.py --binary_classification\
    --mapper=../../datasets/anger_mapper.json \
    --remove-hashtags=anger,angry,angrryyy,rage \
    --output_filename=anger_dataset.csv \
    ../../datasets/crowdflower/text_emotion.csv \
    ../../datasets/Wang/test.csv
