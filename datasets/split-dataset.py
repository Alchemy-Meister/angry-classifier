# /usr/bin/env
# -*- coding: utf-8 -*-

import getopt
import numpy as np
import os
import pandas as pd
import sys
import ujson

CSV_COLUMNS = ['tweet_id', 'label', 'author', 'content']

# Author column is not required.
COMPULSORY_COLUMNS = list(CSV_COLUMNS)
del COMPULSORY_COLUMNS[2]

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CWD = os.getcwd()
OUTPUT_DIR = os.path.join(DIR_PATH, 'splitted')
BINARY_TAG = 'binary_'
OUTPUT_FILENAMES = ['splitted_dataset_1.csv', 'splitted_dataset_2.csv']

NON_ENGLISH_CHARS = ('¿','À','Á','Â','Ã','Ä','Å','Æ','Ç','È','É','Ê','Ë','Ì', \
    'Í','Î','Ï','Ð','Ñ', 'Ò','Ó','Ô','Õ','Ö','×','Ø','Ù','Ú','Û','Ü','Ý','Þ', \
    'ß','à','á','â','ã','ä','å','æ','ç','è','é','ê','ë','ì','í','î','ï','ð', \
    'ñ','ò','ó','ô','õ','ö','ø','ù','ú','û','ü','ý','þ','ÿ')

def path_num_checker(path_num):
    if path_num < 2:
        print 'ERROR: at least two dataset paths must be provided.'
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
    check_valid_dir(OUTPUT_DIR)

    # Flag to perform binary classification.
    binary_classification = False
    mappers = []
    hashtags = []
    size_ratios = [0.5, 0.5]

    try:
        opts, args = getopt.getopt(argv,'b',['binary_classification', \
            'mappers=', 'remove-hashtags=', 'output_filenames=', \
            'size_ratios='])
    except getopt.GetoptError:
        print 'ERROR: Unknown parameter. Usage: split_datasets.py [-b] ' \
            + '[--binary_classification] --size_ratios=0.7,0.15,0.15' \
            + '--mappers=path_to_mapper_1,path_to_mapper_2 ' \
            + '--remove-hashtags=tag1,tag2,tagN '\
            + 'output_filenames=output_name_1.csv,output_name_N.csv ' \
            + 'path_to_dataset'
        sys.exit(2)

    for o, a in opts:
        if o == '-b' or o == '--binary_classification':
            # Activate binary classification flag.
            binary_classification = True
        elif o == '--mappers':
            for mapper in a.split(','):
                if mapper != '':
                    mappers.append(check_valid_path(mapper, 'mapper'))
        elif o == '--output_filenames':
            global  OUTPUT_FILENAME
            OUTPUT_FILENAMES = []
            for output_filename in a.split(','):
                if output_filename != '':
                    OUTPUT_FILENAMES.append(output_filename)
        elif o == '--remove-hashtags':
            for hashtag in a.split(','):
                if hashtag != '':
                    hashtags.append('#' + hashtag)
        elif o == '--size_ratios':
            size_ratios = []
            for size_ratio in a.split(','):
                if size_ratio != '':
                    size_ratios.append(size_ratio)

    if len(args) != 1:
        print 'Error: A unique path to dataset must be provided.'
        sys.exit(2)

    dataset_path = check_valid_path(args[0], 'dataset')

    df = pd.read_csv(dataset_path, header=0, dtype={'tweet_id': np.int64})

    for column in COMPULSORY_COLUMNS:
        if column not in df.columns:
            print 'Error: Invalid dataset was provided. The csv files need ' \
            + 'contain the following columns: %s' % str(COMPULSORY_COLUMNS)
            sys.exit(2)

    # Remove some non-English tweets.
    for non_en_char in NON_ENGLISH_CHARS:
        df = df.drop(df[df.content.str.contains(non_en_char)].index)

    for index, mapper_path in enumerate(mappers):
        mappers[index] = ujson.load(open(mapper_path,'r'))

    if binary_classification:
        if len(mappers) == 0:
            print 'Error: Unable to split into a binary datasets without' \
                + 'the mapper.'
            sys.exit(2)

        mappers_len = len(mappers)

        positive_tweets_dfs = [None] * mappers_len
        positive_labels = []
        remove_labels = [None] * mappers_len


        positive_tweets_number = 0
        positive_tweets_length = [0] * mappers_len

        for index, mapper in enumerate(mappers):
            binary_label = mapper['binary']
            positive_labels.append(binary_label)
            remove_labels[index] = mapper['remove']

            # Creates a binary sub-samples of the original dataframe
            positive_tweets_dfs[index] = df[(df.label == binary_label)].copy()

            current_pos_tweets_length = len(positive_tweets_dfs[index].index)

            positive_tweets_number += current_pos_tweets_length
            positive_tweets_length[index] = current_pos_tweets_length

        del current_pos_tweets_length

        negative_tweets_df = df[ \
            ~df[COMPULSORY_COLUMNS[1]].isin(positive_labels)].copy()

        del df

        if len(negative_tweets_df.index) < positive_tweets_number:
            print 'Error: Could\'t create normalized binary datasets,' \
                + 'not enough negative tweets.'
            sys.exit(2)

        hashtag_num = len(hashtags)

        # Remove target labels (hashtags) from tweet content.
        for hastag in hashtags:
            negative_tweets_df[CSV_COLUMNS[3]] = negative_tweets_df[ \
                CSV_COLUMNS[3] ].str.replace(hastag, ' ', case=False)

        # Trim tweet text if needed.
        if hashtag_num > 0:
            negative_tweets_df[CSV_COLUMNS[3]] = negative_tweets_df[ \
                CSV_COLUMNS[3] ].str.strip()

        negative_tweets_dfs = [None] * mappers_len

        for index in xrange(len(mappers)):
            if remove_labels[index] != None:
                negative_tweets_dfs[index] = negative_tweets_df[ \
                    ~negative_tweets_df[CSV_COLUMNS[1]] \
                    .isin(remove_labels[index])].copy()

                negative_tweets_dfs[index][CSV_COLUMNS[1]] = 'no_' \
                    + positive_labels[index]

                print negative_tweets_dfs[index]


if __name__ == "__main__":
    main(sys.argv[1:])