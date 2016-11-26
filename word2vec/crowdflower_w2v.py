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
from nltk.tokenize import RegexpTokenizer
import numpy as np
import os
import pandas as pd
import re
import string
import sys
from tqdm import trange, tqdm
import ujson

# Hack to import modules form sibling paths.
sys.path.insert(0, os.path.abspath('..'))

import preprocessing.spelling_corrector.spell as spell

start_time = datetime.now()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_PATH = './../datasets/crowdflower/'
DATASET_FILENAME = 'text_emotion.csv'
OUTPUT_PATH = 'json/crowdflower'
DISTRIBUTION = '_distribution'

NUM_MODEL_FEATURES = 300

def process_sample(model, sample, list_index, max_word_per_sentence):
    num_words = len(sample[list_index]['words'])

    for index in xrange(max_word_per_sentence):

        if index < num_words:

            word = sample[list_index]['words'][index]

            try:
                sample[list_index]['word2vec'].append( \
                    model[word].tolist())
            except KeyError:
                # TODO Check the spelling in a dictionary.
                sample[list_index]['word2vec'].append( \
                    np.zeros(NUM_MODEL_FEATURES).tolist())
        else:
            # Adds zero array padding until filling max_word_per_sentence.
            sample[list_index]['word2vec'].append( \
                np.zeros(NUM_MODEL_FEATURES).tolist())

    # Remove word list from the dict
    sample[list_index].pop('words', None)

def serialize_sample(sample_output_path, sample, indent):
     with codecs.open(sample_output_path, 'w', \
        encoding='utf-8') as file:

        if indent:
            file.write(ujson.dumps(sample, indent=4))
        else:
            file.write(ujson.dumps(sample))

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
    stop_words = set(stopwords.words("english"))

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
    distribution['classes'] = {}

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

        emoticon_str = (ur'(\<[\/\\]?3|[\(\)\\|\*\$][\-\^]?[\:\;\=]' \
            ur'|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)')

        emoticon_regex = re.compile(emoticon_str)

        # Lowercase tweet text.
        preprocessed_tweet = preprocessed_tweet.lower()

        # Remove URLs and mentions with representative key code.
        preprocessed_tweet = re.sub(twitter_url_regex, 'URL', \
            preprocessed_tweet)
        preprocessed_tweet = re.sub(twitter_mention_regex, ' MENTION', \
            preprocessed_tweet)

        matches = re.finditer(emoticon_regex, preprocessed_tweet)

        for match_num, match in enumerate(matches):
            # print match.group()
            # TODO Compare text file.
            # SAVE data into the model.
            
            break;

        # Removes all punctuation, contraction included.
        # preprocessed_tweet =  preprocessed_tweet.encode('utf-8') \
        #     .translate(None, string.punctuation)
        # preprocessed_tweet.decode('utf-8')
        #
        # Trims generated string.
        # tweet_words = preprocessed_tweet.strip()

        # Removes punctuation, except contractions.
        # tweet_words = [word.strip(string.punctuation) \
        #    for word in preprocessed_tweet.split()]

        # Removes punctuation, contraction included and separate them.
        tokenizer = RegexpTokenizer(r'\w+')
        tweet_words = tokenizer.tokenize(preprocessed_tweet)

        for word_index in xrange(len(tweet_words)):
            word = tweet_words[word_index]

            if not word.isdigit():
                corrected_word = spell.correction(word)
                if word != corrected_word:
                    print word + ': ' + corrected_word
                    tweet_words[word_index] = corrected_word


        # Removes stopwords.
        tweet_words = [word for word in tweet_words \
            if word and word not in stop_words]

        word_count = len(tweet_words)

        if divide and preprocess_index >= train_length:
            test_output.append({'label': label, 'words': tweet_words, \
                'word2vec': []});
        elif divide and preprocess_index < train_length:
            output.append({'label': label, 'words': tweet_words, \
                'word2vec': []});
        elif not divide:
            output.append({'label': label, 'words': tweet_words, \
                'word2vec': []});

        # Update largest sentence's word number.
        if max_word_per_sentence < word_count:
            max_word_per_sentence = word_count

        # Calculates dataset class distribution.
        if label in distribution['classes']:
            distribution['classes'][label] += 1
        else:
            distribution['classes'][label] = 1

        # Update index.
        preprocess_index += 1

    distribution['max_phrase_length'] = max_word_per_sentence
    distribution['model_feature_length'] = NUM_MODEL_FEATURES

    for train_index in trange(len(output), desc=task_description, \
            total=len(output)):

        process_sample(model, output, train_index, max_word_per_sentence)

    # Optional if statement to hide console progress bar when not needed.
    if divide:
        for test_index in trange(len(test_output), \
            desc='Calculating Word2Vec test values', total=len(test_output)):

            process_sample(model, test_output, test_index, \
                max_word_per_sentence)

    serialization_start_time = datetime.now()

    if divide:
        logger.info('Serializing JSON into train and test files.')

        # Write train data to a JSON file.
        serialize_sample(DATASET_PATH + OUTPUT_PATH + '_train.json', output, \
            indent)

        # Write test data to a JSON file.
        serialize_sample(DATASET_PATH + OUTPUT_PATH + '_test.json', \
            test_output, indent)
    else:

        logger.info('Serializing JSON into a file.')

        # Write data to a JSON file.
        serialize_sample(DATASET_PATH + OUTPUT_PATH + '.json', \
            output, indent)

    with codecs.open(DATASET_PATH + OUTPUT_PATH + DISTRIBUTION + '.json', 'w', \
            encoding='utf-8') as dout:

            dout.write(ujson.dumps(distribution, indent=4))

    logger.info('Serialization finished, elapsed time: %s', \
        str(datetime.now() - serialization_start_time))

    logger.info('Total elapsed time: %s', \
        str(datetime.now() - start_time))

if __name__ == '__main__':
    main(sys.argv[1:])