#!/bin/sh

python ../../datasets/merge_datasets.py \
    --output_filename=multilabel_merged_dataset.csv \
    ../../datasets/crowdflower/text_emotion.csv \
    ../../datasets/Wang/test.csv \
    ../../datasets/Wang/train_2_1.csv \
    ../../datasets/twitter_archiver/irony_ironic_sarcasm.csv
