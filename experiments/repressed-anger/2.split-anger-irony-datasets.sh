#!/bin/sh

python ../../datasets/split-dataset.py --binary_classification \
    --output_filenames=anger_dataset.csv,irony_dataset.csv \
    --mappers=../../datasets/anger_mapper.json,../../datasets/irony_mapper.json \
    --remove-hashtags=anger,angry,angrryyy,rage,annoyed,annoying,irritated,irritating,disgusting,disgusted,frustrated,frustration,frustrating,irony,ironic,sarcasm,sarcastic \
    ../../datasets/merged/multilabel_merged_dataset.csv