# /usr/bin/env
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import codecs
from datetime import datetime
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

start_time = datetime.now()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CWD = os.getcwd()
DISTRIBUTION = '_distribution'

NUM_MODEL_FEATURES = 300

CLEAN_WORD_LIST = ['URL', 'MENTION']

USAGE_STRING = 'Usage: dataset_w2v.py ' \
            + '[-d] [-i] [-v] [-h] [--sample_division] [--indent] ' \
            + '[--validation] [--size=] [--help] path_to_dataset'

def process_sample(model, sample, list_index, max_word_per_sentence):
    num_words = len(sample[list_index]['words'])

    for index in xrange(max_word_per_sentence):

        if index < num_words:

            word = sample[list_index]['words'][index]

            vector = word2vector(model, word)

            sample[list_index]['word2vec'].append(vector)

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

def word2vector(model, word):
    vector = np.zeros(NUM_MODEL_FEATURES).tolist()

    try:
        vector = model[word].tolist()
    except KeyError:
        if not word.isdigit() and word not in CLEAN_WORD_LIST:
    
            from string import printable

            corrected_word = spell.correction(word)
            if word != corrected_word:
                try:
                    vector = model[corrected_word].tolist()
                except KeyError:
                    pass

    return vector

def main(argv):

    divide = False
    indent = False
    validation = False
    size = None
    task_description = 'Calculating Word2Vec values'

    output_path = None

    try:
        opts, args = getopt.getopt(argv,'divh',['sample_division', 'indent', \
            'size=', 'help'])
    except getopt.GetoptError:
        print 'ERROR: Unknown parameter. %s' % USAGE_STRING 
        sys.exit(2)

    for o, a in opts:
        if o == '-h' or o == '--help':
            print USAGE_STRING
            exit(0)
        elif o == '-d' or o == '--sample_division':
            divide = True
            task_description = 'Calculating Word2Vec train values'
            task2_description = 'Calculating Word2Vec test values'
        elif o == '-i' or o == '--indent':
            indent = True
        elif o == '-v' or o == '--validation':
            divide = True
            task_description = 'Calculating Word2Vec train values'
            task2_description = 'Calculating Word2Vec validation values'
            validation = True
        elif o == '--size':
            try:
                size = int(a)
            except:
                print 'Error: size argument must be an integer.'

    if len(args) != 1:
        print 'Error: Dataset path must be provided.'
        sys.exit(2)
    else:
        args = args[0]

        if not os.path.isabs(args):
            # Make relative path absolute.
            args = os.path.join(CWD, args)

        if os.path.isfile(args):
            # Get dataset dir path.
            source_path = args.rsplit('/', 1)
            dataset_path = source_path[0]

            # Generate output dir path.
            output_path = os.path.join(dataset_path, 'json/' \
                + source_path[1].rsplit('.csv')[0])

            # Loads CSV into dataframe.
            df = pd.read_csv(args, usecols=[1,3], header=0)
        else:
            print 'Error: Invalid dataset file.'
            sys.exit(2)

    # Load heavy modules.
    # Start loading gensim module.
    import gensim
    # Start loading spell corrector.
    global spell
    import preprocessing.spelling_corrector.spell as spell

    # Loads NLTK's stopwords for English.
    stop_words = set(stopwords.words("english"))

    model_load_start_time = datetime.now()

    # Loads Word2Vec model.
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.Word2Vec.load_word2vec_format( \
        './../word2vec/model/GoogleNews-vectors-negative300.bin', binary=True)

    logger.info('Model loading complete, elapsed time: %s', \
        str(datetime.now() - model_load_start_time))

    output = []
    validation_or_test_output = []
    test_output = []
    distribution = {}
    distribution['classes'] = {}

    if size is not None and size <= len(df.index):
        df_length = size
        # Under-size the dataframe to match requested size.
        df = df.take(np.random.permutation(df.index)[:df_length])
    else:
        df_length = len(df.index)

    # Designates 70% of the dataset for training file.
    train_length = ceil(df_length * 0.7)

    # Designates 15% of the dataset for validation and testing.
    if validation:
        test_start = (train_length + (df_length - train_length) / 2 ) - 1

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
        preprocessed_tweet = BeautifulSoup(preprocessed_tweet, 'html.parser') \
            .prettify()

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

        # for match_num, match in enumerate(matches):
        #     print match.group()
        #     # TODO Compare text file.
        #     # SAVE data into the model.
            
        #     break;

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

        # for word_index in xrange(len(tweet_words)):
        #     word = tweet_words[word_index]

        #     try:
        #         vector = model[word].tolist()
        #     except KeyError:
        #         if not word.isdigit() and word not in CLEAN_WORD_LIST:
        #             corrected_word = spell.correction(word)
        #             if word != corrected_word:
        #                 print word + ': ' + corrected_word
        #                 tweet_words[word_index] = corrected_word

        # Removes stopwords.
        tweet_words = [word for word in tweet_words \
            if word and word not in stop_words]

        word_count = len(tweet_words)

        if divide and preprocess_index >= train_length:
            # Split data into validation file.
            if not validation or preprocess_index <= test_start:
                validation_or_test_output.append({'label': label, \
                    'words': tweet_words, 'word2vec': []});
            else:
                # Split data into test validation file.
                test_output.append({'label': label, \
                    'words': tweet_words, 'word2vec': []});
        elif divide and preprocess_index < train_length:
            # Split data into train file.
            output.append({'label': label, 'words': tweet_words, \
                'word2vec': []});
        elif not divide:
            # Adds all the data to a single file, without splitting.
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
        for val_test_index in trange(len(validation_or_test_output), \
            desc=task2_description, total=len(validation_or_test_output)):

            process_sample(model, validation_or_test_output, \
                val_test_index, max_word_per_sentence)

        if validation:
            for test_index in trange(len(test_output), \
                desc='Calculating Word2Vec test values', \
                total=len(test_output)):

                process_sample(model, test_output, test_index, \
                    max_word_per_sentence)

    # Release memory.
    del model
    del df

    serialization_start_time = datetime.now()

    if divide:
        if validation:
            logger.info('Serializing JSON into train, ' \
                + 'validation and test files.')
            
            # Write validation data to a JSON file.
            serialize_sample(output_path + '_validation.json', \
                validation_or_test_output, indent)
            
            # Resealse memory.
            del validation_or_test_output

        else:
            logger.info('Serializing JSON into train and test files.')

        # Write train data to a JSON file.
        serialize_sample(output_path + '_train.json', output, \
            indent)
        
        # Resealse memory.
        del output

        # Write test data to a JSON file.
        serialize_sample(output_path + '_test.json', \
            test_output, indent)
        
        # Resealse memory.
        del test_output

    else:

        logger.info('Serializing JSON into a file.')

        # Write data to a JSON file.
        serialize_sample(output_path + '.json', \
            output, indent)

        # Resealse memory.
        del output

    with codecs.open(output_path + DISTRIBUTION + '.json', 'w', \
            encoding='utf-8') as dout:

        dout.write(ujson.dumps(distribution, indent=4))

    # Release memory.
    del distribution

    logger.info('Serialization finished, elapsed time: %s', \
        str(datetime.now() - serialization_start_time))

    logger.info('Total elapsed time: %s', \
        str(datetime.now() - start_time))

if __name__ == '__main__':
    main(sys.argv[1:])