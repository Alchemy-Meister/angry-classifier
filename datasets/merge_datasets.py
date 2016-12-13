# /usr/bin/env
# -*- coding: utf-8 -*-

import getopt
import numpy as np
import os
import pandas as pd
import sys
import ujson

CSV_COLUMNS = ['tweet_id', 'sentiment', 'author', 'content']

# Author column is not required.
COMPULSORY_COLUMNS = list(CSV_COLUMNS)
del COMPULSORY_COLUMNS[2]

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CWD = os.getcwd()
OUTPUT_DIR = os.path.join(DIR_PATH, 'merged')
BINARY_TAG = 'binary_'
OUTPUT_FILENAME = 'emotion_dataset.csv'

NON_ENGLISH_CHARS = ('¿','À','Á','Â','Ã','Ä','Å','Æ','Ç','È','É','Ê','Ë','Ì', \
    'Í','Î','Ï','Ð','Ñ', 'Ò','Ó','Ô','Õ','Ö','×','Ø','Ù','Ú','Û','Ü','Ý','Þ', \
    'ß','à','á','â','ã','ä','å','æ','ç','è','é','ê','ë','ì','í','î','ï','ð', \
    'ñ','ò','ó','ô','õ','ö','ø','ù','ú','û','ü','ý','þ','ÿ')

def path_num_checker(path_num):
    if path_num < 2:
        print 'ERROR: path to at least two dataset paths must be provided.'
        sys.exit(2)

def main(argv):
    # Flag to perform binary classification.
    binary_classification = False
    mapper = None

    try:
        opts, args = getopt.getopt(argv,'b',['binary_classification', \
            'mapper='])

    except getopt.GetoptError:
        print 'ERROR: Unknown parameter. Usage: merge_datasets.py [-b] ' \
            + '[--binary_classification] --mapper=path_to_mapper ' \
            + 'path_to_dataset_1 path_to_dataset_N-1 path_to_dataset_N'
        sys.exit(2)

    for o, a in opts:
        if o == '-b' or o == '--binary_classification':
            # Activate binary classification flag.
            binary_classification = True
        elif o == '--mapper':
            # Make relative path absolute.
            if not os.path.isabs(a):
                a = os.path.join(CWD, a)

            if os.path.isfile(a):
                mapper = ujson.load(open(a,'r'))
            else:
                print 'ERROR: Invalid mapper path.'
                sys.exit(2)

    num_valid_paths = len(args)

    # Check if minimum number of path is fulfilled
    path_num_checker(num_valid_paths)

    dataset = None

    for dataset_path in args:
        if not os.path.isabs(dataset_path):
            # Make relative path absolute.
            dataset_path = os.path.join(CWD, dataset_path)

        if os.path.isfile(dataset_path):
            # Loads CSV into dataframe.
            df = pd.read_csv(dataset_path, header=0, \
                dtype={'tweet_id': np.int64})

            valid_dataset = True

            # Check if all columns exist on dataset.
            for column in COMPULSORY_COLUMNS:
                if column not in df.columns:
                    # Not a valid dataset file.
                    num_valid_paths -= 1
                    path_num_checker(num_valid_paths)
                    valid_dataset = False
                    break

            if valid_dataset:
                # Reorder dataframe's columns
                df = df[CSV_COLUMNS]

                # Remove some non-English tweets.
                for non_en_char in NON_ENGLISH_CHARS:
                    df = df.drop(df[df.content.str.contains(non_en_char)] \
                        .index)

                # Append current dataframe to global.
                if dataset is None:
                    dataset = df
                else:
                    dataset = dataset.append(df, ignore_index=True)\
                        .drop_duplicates(keep=False)

        else:
            # Not a valid dataset path.
            num_valid_paths -= 1
            path_num_checker(num_valid_paths)

    if mapper is not None:
        # Replace sentiment name based on mapper.
        for key, value in mapper['map'].iteritems():
            dataset['sentiment'] = dataset['sentiment'].replace(key, value)
        # Remove sentiment types based on mapper.
        for value in mapper['remove']:
            dataset = dataset.drop(dataset[dataset.sentiment == value].index)

    if binary_classification:
        # Creates a binary sub-samples of the original dataframe
        anger_tweets = dataset[(dataset.sentiment == 'anger')].copy()
        no_anger_tweets = dataset[(dataset.sentiment != 'anger')].copy()

        # Replace sentiment labels for binary classification
        anger_tweets['sentiment'] = 'anger'
        no_anger_tweets['sentiment'] = 'no_anger'

        # Select a random subset of len(anger_tweets) without replacement.
        no_anger_tweets = no_anger_tweets.take( \
            np.random.permutation(len(no_anger_tweets))[:len(anger_tweets)] )

        # Merge both dataframes.
        anger_tweets = anger_tweets.append(no_anger_tweets, ignore_index=True)

        # Shuffle
        dataset = anger_tweets.sample(frac=1).reset_index(drop=True)

        # Serialize dataframe.
        dataset.to_csv(path_or_buf=os.path.join(OUTPUT_DIR, \
            BINARY_TAG + OUTPUT_FILENAME), \
            header=['tweet_id', 'sentiment', 'author', 'content'], index=False,
            encoding='utf-8')
    else:
        #Serialize dataframe.
        dataset.to_csv(path_or_buf=os.path.join(OUTPUT_DIR, OUTPUT_FILENAME), \
            header=['tweet_id', 'sentiment', 'author', 'content'], index=False,
            encoding='utf-8')

if __name__ == "__main__":
    main(sys.argv[1:])