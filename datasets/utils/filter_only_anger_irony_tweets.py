# /usr/bin/env
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys

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

def main(argv):
    csv_name = argv[0].split('.csv')[0].rsplit('/', 1)[1]
    print csv_name
    df = pd.read_csv(check_valid_path(argv[0], 'csv path'), header=0)
    df = df[(df.label == 'anger') | (df.label == 'irony')]

    # Serializes prediction CSV
    df.to_csv(path_or_buf=os.path.join(CWD, csv_name + '.csv'), \
        index=False, encoding='utf-8')

if __name__ == '__main__':
    main(sys.argv[1:])