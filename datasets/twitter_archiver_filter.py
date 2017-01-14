# /usr/bin/env
# -*- coding: utf-8 -*-

from datetime import datetime
import getopt
import numpy as np
import pandas as pd
import os, sys
from tqdm import trange, tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'twitter_archiver/')

COMPULSORY_COLUMNS = ['Tweet ID', 'label', 'Screen Name', 'Tweet Text']
RENAME_COLUMS = ['tweet_id', 'label', 'author', 'content']

USAGE_STRING = 'Usage: twitter_archiver_filter.py [-h] [--help]' \
    '[--hashtags=tag1,tag2,tagN] dataset_path'

def path_provided_checker(path_num):
    if path_num != 1:
        print 'ERROR: path to a single dataset must be provided.'
        sys.exit(2)

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
    hashtags = []

    # Checks if the output dir exist or create it otherwise
    check_valid_dir(OUTPUT_DIR)

    try:
        opts, args = getopt.getopt(argv,'h',['help', 'hashtags='])
    except getopt.GetoptError:
        print 'Error: Unknown parameter. %s' % USAGE_STRING
        sys.exit(2)

    for o, a in opts:
        if o == '-h' or o == '--help':
            print USAGE_STRING
            sys.exit(0)
        elif o == '--hashtags':
            for hashtag in a.split(','):
                if hashtag != '':
                    hashtags.append('#' + hashtag)

    # Check if dataset path is provided.
    path_provided_checker(len(args))

    dataset_path = check_valid_path(args[0], 'dataset')

    OUTPUT_FILENAME = dataset_path.rsplit('/', 1)[1].split('raw_')[1]

    df = pd.read_csv(dataset_path, header=0, usecols=[1,3,4], \
        dtype={'Tweet ID': np.int64})

    # Remove duplicate tweet texts.
    df = df.drop_duplicates(COMPULSORY_COLUMNS[3], keep='first')

    # Add label column to dataframe.
    df['label'] = pd.Series('irony', index=df.index)
    # Reorder the columns.
    df = df[COMPULSORY_COLUMNS]
    
    df[COMPULSORY_COLUMNS[2]] = df[COMPULSORY_COLUMNS[2]].str.replace('@', '')
    
    hashtag_num = len(hashtags)

    # Remove target labels (hashtags) from tweet content.
    for hastag in hashtags:
        df[COMPULSORY_COLUMNS[3]] = df[COMPULSORY_COLUMNS[3]].str \
            .replace(hastag, ' ', case=False)

    # Trim tweet text if needed.
    if hashtag_num > 0:
        df[COMPULSORY_COLUMNS[3]] = df[COMPULSORY_COLUMNS[3]].str \
            .strip()


    df.to_csv(path_or_buf=os.path.join(OUTPUT_DIR, OUTPUT_FILENAME), \
            header=RENAME_COLUMS, index=False,
            encoding='utf-8')

if __name__ == '__main__':
    start_time = datetime.now()
    main(sys.argv[1:])
    print "Elapsed time: " + str(datetime.now() - start_time)
