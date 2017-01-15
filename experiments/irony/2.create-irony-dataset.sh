#!/bin/sh

python ../../datasets/merge_datasets.py --binary_classification\
    --mapper=../../datasets/irony_mapper.json \
    --output_filename=irony_dataset.csv \
    ../../datasets/twitter_archiver/irony_ironic_sarcasm.csv \
    ../../datasets/crowdflower/text_emotion.csv \
    ../../datasets/Wang/test.csv
