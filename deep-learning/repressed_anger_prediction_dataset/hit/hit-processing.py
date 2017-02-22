# /usr/bin/env
# -*- coding: utf-8 -*-

import getopt
import itertools as it
import numpy as np
import operator
import os
import pandas as pd
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

FORM_SIZE = 100

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

    classification_idxs = []
    irony_idxs = []

    files = []

    responses = {}

    for file in os.listdir(SCRIPT_DIR):
        if file.endswith('.csv'):
            files.append(file)

    files = sorted(files)

    for file in files:
        # Loads CSV into dataframe.
        df = pd.read_csv(os.path.join(SCRIPT_DIR, file))
        
        # Removes the timestamp column from the dataframe.
        df.drop(REMOVE_COLUMN, axis=1, inplace=True)

        # Split the dataframe into the classification and irony questions.
        classification_df = df.filter(regex='.*' + CLASSIFICATION_COLUMN \
            + '.*$', axis=1)
        irony_df = df.filter(regex='.*' + IRONY_COLUMN + '.*$', axis=1)

        try:
            part_and_responses_str = file.split('out')[0].split()
            part_start_idx = int(part_and_responses_str[-2][:-1])
            responses[part_start_idx] = int(part_and_responses_str[-1])
            part_start_idx = part_start_idx * FORM_SIZE - FORM_SIZE

        except:
            print 'Error, invalid file name format. Eg: Name (Part 1) 5 out '\
                + 'of 5 responses.csv'
            sys.exit(2)

        # Get the tweet original position for the irony questions.
        irony_idx = [part_start_idx + df.columns.get_loc(column) \
            for column in irony_df.columns ]

        # Fix the deviation of irony index.
        for index, idx in enumerate(irony_idx):
            irony_idx[index] -= 1 + index

        # Append current file's index to an array that contains all files'
        classification_idxs.extend(range(part_start_idx, part_start_idx \
            + FORM_SIZE))
        irony_idxs.extend(irony_idx)

        df_list = [classification_df, irony_df]

        # Generates frequencies for classification and irony dataframes.
        for index in range(len(df_list)):
            df_result_list[index] = df_result_list[index].append( \
                generate_frequency_df(column_list[index], df_list[index]), \
                ignore_index=True )

    idxs_list = [classification_idxs, irony_idxs]
    remove_tweet_count = 0

    for index, df in enumerate(df_result_list):
        # Change frequency values from float to int.
        df[column_list[index]] = df[column_list[index]].astype(int)
        
        # The max value for each column.
        rowmax = df.max(axis=1)
        # Boolean NP array stating which cells are maximum.
        idx = np.where(df.values == rowmax[:,None])

        # Magic to list all the instances that are maximum for each column.
        groups = it.groupby(zip(*idx), key=operator.itemgetter(0))
        max_class = [[df.columns[j] for i, j in grp] for k, grp in groups]

        df_class = pd.DataFrame(columns=result_column_list[index])

        # Deal with multiple draw maximum responses.
        for max_class_index, classification in enumerate(max_class):
            max_length = len(classification)
            max_label = {}
            # If there's no draw & the maximum answer has more than half votes.
            if max_length == 1 and rowmax[max_class_index] > responses[ \
                int(idxs_list[index][max_class_index] / FORM_SIZE) + 1] / 2:

                # Insert the maximum value as it is.
                max_label[result_column_list[index][0]] = classification[0]

            else:
                # If there's draw, insert key value as later removal flag.
                max_label[result_column_list[index][0]] = 'REMOVE'
                remove_tweet_count = remove_tweet_count + 1

            df_class = df_class.append(max_label, ignore_index=True)

        df_result_list[index] = df_class

    print 'Ambiguous tweets: %s' % remove_tweet_count

    # Change Irony y/n response columns to target labels.
    for index, label in enumerate(IRONY_COLUMNS):
        df_result_list[1][df_result_list[1][IRONY_RESULT_COLUMN[0]] == label] \
            = ORIGINAL_IRONY_LABELS[index]

    # Load original dataset.
    df = pd.read_csv(dataset_path, header=0, \
        dtype={COMPULSORY_COLUMNS[0]: np.int64})

    for index, dataframe in enumerate(df_result_list):
        # Update dataframe index to original row position values.
        df_result_list[index].index = idxs_list[index]

    # Add manual classification and irony columns.
    df[CLASSIFICATION_RESULT_COLUMN[0]] = np.nan
    df[IRONY_RESULT_COLUMN[0]] = np.nan

    # Add manual classification results to proper rows.
    for row in df_result_list[0].itertuples():
        df.ix[row[0], 4] = row[1]

    # Add manual irony results to proper rows.
    for row in df_result_list[1].itertuples():
        df.ix[row[0], IRONY_RESULT_COLUMN[0]] = row[1]

    # Fixes irony classified target label, according to manual irony column.
    for row in df[ (df[COMPULSORY_COLUMNS[1]] != df[IRONY_RESULT_COLUMN[0]]) \
        & (df[IRONY_RESULT_COLUMN[0]].notnull()) ].itertuples():

        df.ix[row[0], COMPULSORY_COLUMNS[1]] = row[len(row) - 1]

    # Removes the manual irony column from the dataframe.
    df.drop(IRONY_RESULT_COLUMN[0], axis=1, inplace=True)

    # Removes all the rows that have the revmoval flag.
    df = df[ (~df[SERIALIZE_COLUMNS[4]].str.match('REMOVE', na=False)) & \
        (~df[SERIALIZE_COLUMNS[1]].str.match('REMOVE', na=False)) ]

    # Serialize resulting dataframe.
    df.to_csv(path_or_buf=output, header=SERIALIZE_COLUMNS, index=False,
            encoding='utf-8')

if __name__ == '__main__':
    main(sys.argv[1:])