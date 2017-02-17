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
CLASSIFICATION_COLUMNS = [label.lower() for label in LABELS]

IRONY_LABELS = ['Yes', 'No']
IRONY_COLUMNS = [label.lower() for label in IRONY_LABELS]


def generate_frequency_df(columns, df):
    result_df = pd.DataFrame(columns=columns)

    for column in df.columns:
        tweet_count = df[column].value_counts()

        tweet_result = {}

        for label in columns:
            tweet_result[label] = 0

        for index, label in enumerate(tweet_count.index):
            tweet_result[label.lower()] = tweet_count[index]

        result_df = result_df.append(tweet_result, ignore_index=True)

    return result_df

def main(argv):
    classification_result_df = pd.DataFrame(columns=CLASSIFICATION_COLUMNS)
    irony_result_df = pd.DataFrame(columns=IRONY_COLUMNS)

    df_result_list = [classification_result_df, irony_result_df]
    column_list = [CLASSIFICATION_COLUMNS, IRONY_COLUMNS]


    for file in os.listdir(SCRIPT_DIR):
        if file.endswith('.csv'):
            # print file
            # Loads CSV into dataframe.
            df = pd.read_csv(os.path.join(SCRIPT_DIR, file))
            df.drop(REMOVE_COLUMN, axis=1, inplace=True)

            #print df
            #print df.columns

            classification_df = df.filter(regex='.*' + CLASSIFICATION_COLUMN \
                + '.*$', axis=1)

            irony_df = df.filter(regex='.*' + IRONY_COLUMN + '.*$', axis=1)

            # print df.columns
            # print classification_df.columns
            # print irony_df.columns

            # Get the tweet original position for the irony column.
            # for column in irony_df.columns:
            #    print df.columns.get_loc(column) - 1

            df_list = [classification_df, irony_df]

            for index in range(len(df_list)):
                df_result_list[index] = df_result_list[index].append( \
                    generate_frequency_df(column_list[index], df_list[index]), \
                    ignore_index=True )


    for index, df in enumerate(df_result_list):
        df[column_list[index]] = df[column_list[index]].astype(int)
        print df

if __name__ == '__main__':
    main(sys.argv[1:])