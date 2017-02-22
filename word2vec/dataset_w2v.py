# /usr/bin/env
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import codecs
from datetime import datetime
import getopt
import HTMLParser
import logging
from math import floor, ceil
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

start_time = datetime.now()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CWD = os.getcwd()

# Hack to import modules form sibling paths.
sys.path.insert(0, DIR_PATH + '/..')

DISTRIBUTION = '_distribution'

NUM_MODEL_FEATURES = 300

CLEAN_WORD_LIST = ['URL', 'MENTION']

USAGE_STRING = 'Usage: dataset_w2v.py ' \
            + '[-d] [-g] [-i] [-v] [-s] [-h] [--sample_division] [--indent] ' \
            + '[--validation] [--spell_check] [--size=] [--split_ratio=] ' \
            + '[--max_phrase_length=] [--google_model] [--delete-hashtags] ' \
            + '[--help] path_to_dataset'

CSV_COLUMNS = ['tweet_id', 'label', 'author', 'content']

# Author column is not required.
COMPULSORY_COLUMNS = list(CSV_COLUMNS)
del COMPULSORY_COLUMNS[2]

LOAD_COLUMNS = list(COMPULSORY_COLUMNS)
del LOAD_COLUMNS[0]

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

def process_sample(model, sample, list_index, max_word_per_sentence, spell):
    num_words = len(sample[list_index]['words'])

    for index in xrange(max_word_per_sentence):

        if index < num_words:

            word = sample[list_index]['words'][index]

            vector = word2vector(model, word, spell)

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

def word2vector(model, word, spell_check):
    vector = np.zeros(NUM_MODEL_FEATURES).tolist()

    try:
        vector = model[word].tolist()
    except KeyError:
        if spell_check and not word.isdigit() and word not in CLEAN_WORD_LIST:
    
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
    delete_hashtags = False
    spell_check = False
    size = None
    force_max_phrase_length = None
    use_twitter_model = False
    task_description = 'Calculating Word2Vec values'

    model_rel_path = '/model/GoogleNews-vectors-negative300.bin'

    split_ratios = []
    split_ratios_lenght = 0
    output_path = None

    try:
        opts, args = getopt.getopt(argv,'divtsh',['sample_division', 'indent', \
            'validation', 'size=', 'split_ratio=', 'spell_check', 'help', \
            'max_phrase_length=', 'delete-hashtags', 'twitter_model'])
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
        elif o == '--max_phrase_length':
            try:
                force_max_phrase_length = int(a)
            except:
                print 'Error: max phrease length argument must be an integer.'
                sys.exit(2)
        elif o == '--size':
            try:
                size = int(a)
            except:
                print 'Error: size argument must be an integer.'
                sys.exit(2)
        elif o == '--split_ratio':
            for split_ratio in a.split(','):
                if split_ratio != '':
                    try:
                        split_ratios.append(float(split_ratio))
                    except Exception, e:
                        print 'Error: invalid split_ratio format. %s' \
                            % USAGE_STRING
                        sys.exit(2)
            split_ratios_lenght = len(split_ratios)
        elif o == '-s' or o == '--spell_check':
            spell_check = True
        elif o == '--delete-hashtags':
            delete_hashtags = True
        elif o == '-t' or o == '--twitter_model':
            use_twitter_model = True
            import word2vecReader
            model_rel_path = '/model/word2vec_twitter_model.bin'
            global NUM_MODEL_FEATURES
            NUM_MODEL_FEATURES = 400

    if split_ratios_lenght != 0:
        if validation and split_ratios_lenght != 3:
            print 'Error: 3 split ratios must be provided on validation.'
            sys.exit(2)
        elif divide and not validation and split_ratios_lenght != 2:
            print 'Error: 2 split ratios must be provided on sample division.'
            sys.exit(2)

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
            check_valid_dir(os.path.join(dataset_path, 'json/'))
            output_path = os.path.join(dataset_path, 'json/' \
                + source_path[1].rsplit('.csv')[0])

            if spell_check:
                output_path = output_path + '_spell'

            # Loads CSV into dataframe.
            df = pd.read_csv(args, header=0, usecols=LOAD_COLUMNS)
        else:
            print 'Error: Invalid dataset file.'
            sys.exit(2)

    # Load heavy modules.
    # Start loading gensim module.
    import gensim
    # Start loading spell corrector.
    if spell_check:
        global spell
        import preprocessing.spelling_corrector.spell as spell

    # Loads NLTK's stopwords for English.
    stop_words = set(stopwords.words("english"))

    model_load_start_time = datetime.now()

    # Loads Word2Vec model.
    # Load Google's pre-trained Word2Vec model.
    if use_twitter_model:
        model = model = word2vecReader.Word2Vec.load_word2vec_format(DIR_PATH \
            + model_rel_path, binary=True)
    else:
        model = gensim.models.Word2Vec.load_word2vec_format( \
            DIR_PATH + model_rel_path, binary=True)

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

    if split_ratios_lenght == 0:
        # Designates 70% of the dataset for training file.
        train_length = ceil(df_length * 0.7)

        # Designates 15% of the dataset for validation and testing.
        if validation:
            test_start = (train_length + (df_length - train_length) / 2 ) - 1
    else:
        train_length = round(df_length * split_ratios[0]) -1

        if validation:
            test_start = train_length \
                + floor(df_length * split_ratios[1]) - 1

    max_word_per_sentence = 0

    preprocess_index = 0

    # Twitter preprocessing: replacing URLs and Mentions
    twitter_url_str = (ur'http[s]?://(?:[a-zA-Z]|[0-9]|' \
        ur'[$+*%/@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    twitter_mention_regex = re.compile(ur'(\s+|^|\.)@\S+')
    twitter_hashtag_regex = re.compile(ur'(\s+|^|\.)#\S+')
    twitter_url_regex = re.compile(twitter_url_str)

    emoticon_str = (ur'(\<[\/\\]?3|[\(\)\\|\*\$][\-\^]?[\:\;\=]' \
            ur'|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)')

    emoticon_regex = re.compile(emoticon_str)

    for tweet in tqdm(df.itertuples(), desc='Tweet preprocessing', \
        total=df_length):

        label = tweet[1]

        # Clean text of new lines.
        preprocessed_tweet = tweet[2].replace('\n', ' ');

        # Unescape possible HTML entities.
        preprocessed_tweet = BeautifulSoup(preprocessed_tweet, 'html.parser') \
            .prettify()

        # Lowercase tweet text.
        preprocessed_tweet = preprocessed_tweet.lower()

        # Remove URLs and mentions with representative key code.
        preprocessed_tweet = re.sub(twitter_url_regex, 'URL', \
            preprocessed_tweet)
        preprocessed_tweet = re.sub(twitter_mention_regex, ' MENTION', \
            preprocessed_tweet)

        if delete_hashtags:
            preprocessed_tweet = re.sub(twitter_hashtag_regex, 'TAG', \
                preprocessed_tweet)

        # Search for emoticons.
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

    del twitter_url_regex
    del twitter_mention_regex
    del twitter_hashtag_regex
    del emoticon_regex

    if force_max_phrase_length != None:
        max_word_per_sentence = force_max_phrase_length

    distribution['max_phrase_length'] = max_word_per_sentence
    distribution['model_feature_length'] = NUM_MODEL_FEATURES

    for train_index in trange(len(output), desc=task_description, \
        total=len(output)):

        process_sample(model, output, train_index, max_word_per_sentence, \
            spell_check)

    # Optional if statement to hide console progress bar when not needed.
    if divide:
        for val_test_index in trange(len(validation_or_test_output), \
            desc=task2_description, total=len(validation_or_test_output)):

            process_sample(model, validation_or_test_output, \
                val_test_index, max_word_per_sentence, spell_check)

        if validation:
            for test_index in trange(len(test_output), \
                desc='Calculating Word2Vec test values', \
                total=len(test_output)):

                process_sample(model, test_output, test_index, \
                    max_word_per_sentence, spell_check)

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