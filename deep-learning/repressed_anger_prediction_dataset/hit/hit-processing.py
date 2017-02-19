# /usr/bin/env
# -*- coding: utf-8 -*-

import getopt
import itertools as it
import numpy as np
import operator
import os
import pandas as pd
from random import randint
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()

REMOVE_COLUMN = 'Timestamp'
CLASSIFICATION_COLUMN = 'Classify this sentence as'
IRONY_COLUMN = 'Does this sentence contains irony?'

LABELS = ['Explicit anger', 'Repressed anger', 'Normal', 'Irony']
CLASSIFICATION_COLUMNS = [label.lower() for label in LABELS]

ORIGINAL_IRONY_LABELS = ['irony', 'no_irony']
IRONY_LABELS = ['Yes', 'No']
IRONY_COLUMNS = [label.lower() for label in IRONY_LABELS]

CLASSIFICATION_RESULT_COLUMN = ['manual_label']
IRONY_RESULT_COLUMN = ['manual_irony']

CSV_COLUMNS = ['tweet_id', 'label', 'author', 'content']

# Author column is not required.
COMPULSORY_COLUMNS = list(CSV_COLUMNS)
del COMPULSORY_COLUMNS[2]

SERIALIZE_COLUMNS = CSV_COLUMNS
SERIALIZE_COLUMNS.extend(CLASSIFICATION_RESULT_COLUMN)

def check_valid_path(path, desc):
    if not os.path.isabs(path):
        # Make relative path absolute.
        path = os.path.join(CWD, path)

    if not os.path.isfile(path):
        print 'Error: Invalid %s file.' % desc
        sys.exit(2)

    return path

def check_valid_dir(dir_name):
    if not os.path.isabs(dir_name):
        dir_name = os.path.join(CWD, dir_name)

    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
        except:
            print 'Error: Output directory file could not be created.'
            print dir_name
            sys.exit(2)

    return dir_name

def generate_frequency_df(columns, df):
    result_df = pd.DataFrame(columns=columns)

    # Every column of the df represents a question of the form.
    for column in df.columns:
        # Calculates the frequencies for the current column from all rows.
        tweet_count = df[column].value_counts()

        tweet_result = {}

        # Initializes a dict with all the labels sets as zero.
        for label in columns:
            tweet_result[label] = 0

        # Sets the real count for the frequencies for the current question.
        for index, label in enumerate(tweet_count.index):
            tweet_result[label.lower()] = tweet_count[index]

        # Append the current question's frequencies to the dataset.
        result_df = result_df.append(tweet_result, ignore_index=True)

    return result_df

def main(argv):
    if(len(argv) != 1):
        print 'Error, original dataset csv file path is required.'
        sys.exit(2)

    dataset_path = check_valid_path(argv[0], 'original csv')

    output = dataset_path.split('.csv')[0] + '_hit_processed.csv'

    classification_result_df = pd.DataFrame(columns=CLASSIFICATION_COLUMNS)
    irony_result_df = pd.DataFrame(columns=IRONY_COLUMNS)

    df_result_list = [classification_result_df, irony_result_df]
    column_list = [CLASSIFICATION_COLUMNS, IRONY_COLUMNS]
    result_column_list = [CLASSIFICATION_RESULT_COLUMN, IRONY_RESULT_COLUMN]

    irony_idxs = []

    for file_index, file in enumerate(os.listdir(SCRIPT_DIR)):
        if file.endswith('.csv'):
            # Loads CSV into dataframe.
            df = pd.read_csv(os.path.join(SCRIPT_DIR, file))
            
            # Removes the timestamp column from the dataframe.
            df.drop(REMOVE_COLUMN, axis=1, inplace=True)

            # Split the dataframe into the classification and irony questions.
            classification_df = df.filter(regex='.*' + CLASSIFICATION_COLUMN \
                + '.*$', axis=1)
            irony_df = df.filter(regex='.*' + IRONY_COLUMN + '.*$', axis=1)

            # Get the tweet original position for the irony questions.
            irony_idx = [file_index * 100 + df.columns.get_loc(column) \
                for column in irony_df.columns ]

            # Fix the deviation of irony index.
            for index, idx in enumerate(irony_idx):
                irony_idx[index] -= 1 + index

            # Append current file's index to an array that contains all files'
            irony_idxs.extend(irony_idx)

            df_list = [classification_df, irony_df]

            # Generates frequencies for classification and irony dataframes.
            for index in range(len(df_list)):
                df_result_list[index] = df_result_list[index].append( \
                    generate_frequency_df(column_list[index], df_list[index]), \
                    ignore_index=True )


    for index, df in enumerate(df_result_list):
        # Change frequency values from float to int.
        df[column_list[index]] = df[column_list[index]].astype(int)
        
        # The max value for each column.
        rowmax = df.max(axis=1)
        # Boolean NP array stating which cells are maximum.
        idx = np.where(df.values == rowmax[:,None])

        # Magic to list all the instances that are maximum for each column.
        groups = it.groupby(zip(*idx), key=operator.itemgetter(0))
        rowmax = [[df.columns[j] for i, j in grp] for k, grp in groups]

        df = pd.DataFrame(columns=result_column_list[index])

        # Deal with multiple draw maximum responses.
        for classification in rowmax:
            max_length = len(classification)
            max_label = {}
            # If there's no draw.
            if max_length == 1:
                # Insert the maximum value as it is.
                max_label[result_column_list[index][0]] = classification[0]
            else:
                # If there's draw, select randomly from maximum values.
                max_label[result_column_list[index][0]] = classification[ \
                    randint(0, max_length - 1) ]

            df = df.append(max_label, ignore_index=True)

        df_result_list[index] = df

    # Change Irony y/n response columns to target labels.
    for index, label in enumerate(IRONY_COLUMNS):
        df_result_list[1][df_result_list[1][IRONY_RESULT_COLUMN[0]] == label] \
            = ORIGINAL_IRONY_LABELS[index]

    # Load original dataset.
    df = pd.read_csv(dataset_path, header=0, \
        dtype={COMPULSORY_COLUMNS[0]: np.int64})

    # Add manual classification and irony columns.
    df = pd.concat([df, df_result_list[0]], ignore_index=True, axis=1)
    df[IRONY_RESULT_COLUMN[0]] = np.nan

    # Update dataframe index to original row position values.
    df_result_list[1].index = irony_idxs

    # Add manual irony results to proper rows.
    for row in df_result_list[1].itertuples():
        df.ix[row[0], IRONY_RESULT_COLUMN[0]] = row[1]

    # Fixes irony classified target label, according to manual irony column.
    for row in df[ (df[1] != df[IRONY_RESULT_COLUMN[0]]) \
        & (df[IRONY_RESULT_COLUMN[0]].notnull()) ].itertuples():

        df.ix[row[0], df.columns[1]] = row[len(row) - 1]

    # Removes the manual irony column from the dataframe.
    df.drop(IRONY_RESULT_COLUMN[0], axis=1, inplace=True)

    # Serialize resulting dataframe.
    df.to_csv(path_or_buf=output, header=SERIALIZE_COLUMNS, index=False,
            encoding='utf-8')

if __name__ == '__main__':
    main(sys.argv[1:])