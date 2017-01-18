#!/bin/sh

python ../../datasets/twitter_archiver_filter.py \
    --hashtags=irony,ironic,sarcasm,sarcastic \
    ../../datasets/twitter_archiver/raw_irony_ironic_sarcasm.csv
