# /usr/bin/env
# -*- coding: utf-8 -*-

import codecs
from datetime import datetime
import gensim
import getopt
import HTMLParser
import logging
from math import ceil
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import string
from tqdm import trange, tqdm
import ujson
import sys

start_time = datetime.now()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = './../datasets/crowdflower/'
DATASET_FILENAME = 'text_emotion.csv'
OUTPUT_PATH = 'json/crowdflower'
DISTRIBUTION = '_distribution'

def main(argv):

    divide = False
    indent = False
    task_description = 'Calculating Word2Vec values'

    try:
        opts, args = getopt.getopt(argv,'di',['sample_division', 'indent'])
    except getopt.GetoptError:
        print 'ERROR: Unknown parameter. Usage: crowdflower_w2v.py' \
            + 'dataset_path [-d] [-i] [--sample_division, --indent]'
        sys.exit(2)

    for o, a in opts:
        if o == '-d' or o == '--sample_division':
            divide = True
            task_description = 'Calculating Word2Vec train values'
        if o == '-i' or o == '--indent':
            indent = True

    # Loads NLTK's stopwords for English.
    stop_words = stopwords.words("english")

    # Loads Word2Vec model.
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.Word2Vec.load_word2vec_format( \
        './../word2vec/model/GoogleNews-vectors-negative300.bin', binary=True)

    logger.info('Model loading complete, elapsed time: %s', \
        str(datetime.now() - start_time))

    # Loads CSV into dataframe.
    df = pd.read_csv(DATASET_PATH + DATASET_FILENAME, usecols=[1,3], header=0)

    output = []
    test_output = []
    distribution = {}

    df_length = len(df.index)
    train_length = ceil(df_length * 0.7)

    max_word_per_sentence = 0

    preprocess_index = 0
    for tweet in tqdm(df.itertuples(), desc='Tweet preprocessing', \
        total=df_length):

        label = tweet[1]

        # Twitter preprocessing: replacing URLs and Mentions
        twitter_url_str = (ur'http[s]?://(?:[a-zA-Z]|[0-9]|' \
            ur'[$+*%/@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

        twitter_mention_regex = re.compile(ur'(\s+|^|\.)@\S+')
        twitter_hashtag_regex = re.compile(ur'(\s+|^|\.)#\S+')
        twitter_url_regex = re.compile(twitter_url_str)

        # Clean text of new lines.
        preprocessed_tweet = tweet[2].replace('\n', ' ');

        # Unescape possible HTML entities.
        preprocessed_tweet = HTMLParser.HTMLParser() \
            .unescape(preprocessed_tweet)

        # Remove URLs and mentions with representative key code.
        preprocessed_tweet = re.sub(twitter_url_regex, 'URL', \
            preprocessed_tweet)
        preprocessed_tweet = re.sub(twitter_mention_regex, ' MENTION', \
            preprocessed_tweet)

        # Removes punctuation including hashtags.
        preprocessed_tweet =  preprocessed_tweet.encode('utf-8') \
            .translate(None, string.punctuation)
        preprocessed_tweet.decode('utf-8')

        # Trims generated string.
        preprocessed_tweet = preprocessed_tweet.strip()

        # Escape \ character.
        # preprocessed_tweet = preprocessed_tweet.replace('\\', '\\\\')

        # Generates a list of words
        tweet_words = preprocessed_tweet.split()

        # Removes stopwords.
        tweet_words = [word for word in tweet_words if word not in stop_words]
        word_count = len(tweet_words)

        if divide and preprocess_index >= train_length:
            test_output.append({'label': label, 'words': tweet_words, \
                'word2vec': []});
        elif preprocess_index < train_length:
            output.append({'label': label, 'words': tweet_words, \
                'word2vec': []});

        # Update largest sentence's word number.
        if max_word_per_sentence < word_count:
            max_word_per_sentence = word_count

        # Calculates dataset class distribution.
        if label in distribution:
            distribution[label] += 1
        else:
            distribution[label] = 1

        # Update index.
        preprocess_index += 1

    for index in trange(len(output), desc=task_description, \
            total=len(output)):
        
        for word in output[index]['words']:
            try:
                output[index]['word2vec'].append(model[word].tolist())
            except KeyError:
                # TODO Check the spelling in a dictionary.
                output[index]['word2vec'].append(np.zeros(300).tolist())

        # Remove word list from the dict
        output[index].pop('words', None)

    for index in trange(len(test_output), \
        desc='Calculating Word2Vec test values', total=len(test_output)):

        for word in test_output[index]['words']:
            try:
                test_output[index]['word2vec'].append(model[word].tolist())
            except KeyError:
                # TODO Check the spelling in a dictionary.
                test_output[index]['word2vec'].append(np.zeros(300).tolist())

        # Remove word list from the dict
        test_output[index].pop('words', None)


    serialization_start_time = datetime.now()

    if divide:
        logger.info('Serializing JSON into train and test files.')

        # Write train data to a JSON file.
        with codecs.open(DATASET_PATH + OUTPUT_PATH + '_train.json', 'w', \
            encoding='utf-8') as train_file:

            if indent:
                train_file.write(ujson.dumps(output, indent=4))
            else:
                train_file.write(ujson.dumps(output))

        # Write test data to a JSON file.
        with codecs.open(DATASET_PATH + OUTPUT_PATH + '_test.json', 'w', \
            encoding='utf-8') as test_file:

            if indent:
                test_file.write(ujson.dumps(test_output, indent=4))
            else:
                test_file.write(ujson.dumps(test_output))
    else:

        logger.info('Serializing JSON into a file.')

        # Write data to a JSON file.
        ujson.dump(output, codecs.open(DATASET_PATH + OUTPUT_PATH \
            + '.json', 'w', encoding='utf-8'))

    with codecs.open(DATASET_PATH + OUTPUT_PATH + DISTRIBUTION + '.json', 'w', \
            encoding='utf-8') as dout:

            dout.write(ujson.dumps(distribution, indent=4))

    logger.info('Serialization finished, elapsed time: %s', \
        str(datetime.now() - serialization_start_time))

    logger.info('Total elapsed time: %s', \
        str(datetime.now() - start_time))

if __name__ == '__main__':
    main(sys.argv[1:])