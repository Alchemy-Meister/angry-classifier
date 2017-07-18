# /usr/bin/env
# -*- coding: utf-8 -*-

from collections import OrderedDict
import getopt
import itertools as it
import numpy as np
import operator
import os
import pandas as pd
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()

CSV_COLUMNS = ['tweet_id', 'label', 'author', 'content', 'manual_label']
LABEL_COLUMNS = ['label_1', 'label_2']
OUPUT_COLUMNS = CSV_COLUMNS + LABEL_COLUMNS
LABELS = ['anger', 'no_anger', 'irony', 'no_irony']

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

def main(argv):
    if(len(argv) != 1):
        print 'Error, original dataset csv file path is required.'
        sys.exit(2)

    dataset_path = check_valid_path(argv[0], 'original csv')

    output = dataset_path.split('hit_processed.csv')[0] + 'hit_reprocessed.csv'

    matrix_classes = {
        'explicit anger': ['anger', 'no_irony'],
        'repressed anger': ['anger', 'irony'],
        'normal': ['no_anger', 'no_irony'],
        'irony': ['no_anger', 'irony']
        }

    ordered_matrix_classes = OrderedDict(matrix_classes)
    ordered_matrix_classes = ordered_matrix_classes.keys()

    df = pd.read_csv(dataset_path, header=0, \
        dtype={CSV_COLUMNS[0]: np.int64})

    for column in LABEL_COLUMNS:
        # Adds labels columns to the CSV.
        df[column] = None

    # Sets label columns based on manual classification.
    for index in xrange(len(ordered_matrix_classes)):
        key = ordered_matrix_classes[index]
        value = matrix_classes[key]

        for label_index in xrange(len(LABEL_COLUMNS)):
            df.loc[df[CSV_COLUMNS[4]] == key, LABEL_COLUMNS[label_index]] \
                = value[label_index]

    # Serialize resulting dataframe.
    df.to_csv(path_or_buf=output, header=OUPUT_COLUMNS, index=False,
            encoding='utf-8')

if __name__ == '__main__':
    main(sys.argv[1:])