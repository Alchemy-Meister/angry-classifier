# /usr/bin/env
# -*- coding: utf-8 -*-

import getopt
import numpy as np
import os
import pandas as pd
import sys
import codecs

USAGE_STRING = 'Usage: cnn.py [-h] [--help] --target=[train, test, all] ' \
    + '--train=path_to_train_word2vec ' \
    + '--validation=path_to_validation_word2vect ' \
    + '[--test=path_to_test_word2vec]' \
    + '[--distribution=path_to_distribution_file] ' \
    + '[--load_model=path_to_model] [--load_weights=path_to_weights]'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()

REMOVE_COLUMN = 'Timestamp'
CLASSIFICATION_COLUMN = 'Classify this sentence as'
IRONY_COLUMN = 'Does this sentence contains irony?'

LABELS = ['Explicit anger', 'Repressed anger', 'Normal', 'Irony']
COLUMNS = [label.lower() for label in LABELS]

def main(argv):
    concatenated_result_df = pd.DataFrame(columns=COLUMNS)

    for file in os.listdir(SCRIPT_DIR):
        if file.endswith('.csv'):
            print file
            # Loads CSV into dataframe.
            df = pd.read_csv(os.path.join(SCRIPT_DIR, file))
            df.drop(REMOVE_COLUMN, axis=1, inplace=True)

            #print df
            #print df.columns

            classification_df = df.filter(regex='.*' + CLASSIFICATION_COLUMN \
                + '.*$', axis=1)

            irony_df = df.filter(regex='.*' + IRONY_COLUMN + '.*$', axis=1)

            print df.columns
            print classification_df.columns
            print irony_df.columns

            # Get the tweet original position for the irony column.
            for column in irony_df.columns:
                print df.columns.get_loc(column) - 1

            result_df = pd.DataFrame(columns=COLUMNS)

            for column in classification_df.columns:
                tweet_count = classification_df[column].value_counts()

                tweet_result = {}

                for label in COLUMNS:
                    tweet_result[label] = 0

                for index, label in enumerate(tweet_count.index):
                    tweet_result[label.lower()] = tweet_count[index]

                result_df = result_df.append(tweet_result, ignore_index=True)
                concatenated_result_df = concatenated_result_df.append( \
                    result_df )

    concatenated_result_df[COLUMNS] = concatenated_result_df[COLUMNS] \
        .astype(int)
    print concatenated_result_df



if __name__ == '__main__':
    main(sys.argv[1:])