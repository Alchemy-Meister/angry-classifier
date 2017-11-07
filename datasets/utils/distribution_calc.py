# /usr/bin/env
# -*- coding: utf-8 -*-

import codecs
import os
import pandas as pd
import sys
import ujson

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CWD = os.getcwd()

def check_valid_path(path, desc):
    if not os.path.isabs(path):
        # Make relative path absolute.
        path = os.path.join(CWD, path)

    if not os.path.isfile(path):
        print 'Error: Invalid %s file.' % desc
        print path
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

def process_labels(df, unique_labels, col_name, csv_name):
    distribution = {}
    for label in unique_labels:
        distribution[label] = len(df[df[col_name] == label].index)

    with codecs.open(os.path.join(DIR_PATH, csv_name + '_distribution.json'), \
        'w', encoding='utf-8') as file:

        file.write(ujson.dumps(distribution, indent=4))


def main(argv):
    csv_name = argv[0].split('.csv')[0]
    df = pd.read_csv(check_valid_path(argv[0], 'csv path'), header=0)
    try:
        manual_labels = df.manual_label.unique().tolist()
        process_labels(df, manual_labels, 'manual_label', csv_name)
    except AttributeError:
        automatic_labels = df.label.unique().tolist()
        process_labels(df, automatic_labels, 'label', csv_name)

if __name__ == '__main__':
    main(sys.argv[1:])