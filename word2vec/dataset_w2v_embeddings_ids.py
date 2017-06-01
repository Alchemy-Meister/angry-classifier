# /usr/bin/env
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import codecs
from datetime import datetime
import gc
import getopt
import HTMLParser
from keras.preprocessing.sequence import pad_sequences
import logging
from math import floor, ceil
from mytext import Tokenizer
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

USAGE_STRING = 'Usage: dataset_w2v_embeddings_ids.py ' \
            + '[-d] [-g] [-i] [-v] [-s] [-h] [--sample_division] [--indent] ' \
            + '[--validation] [--spell_check] [--size=] [--split_ratio=] ' \
            + '[--max_phrase_length=] [--twitter_model] [--delete-hashtags] ' \
            + '[--help] --anger_dataset=path_to_dataset ' \
            + '--irony_dataset=path_to_dataset'

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
    try:
        vector = model[word]
    except KeyError:
        if spell_check and not word.isdigit() and word not in CLEAN_WORD_LIST:
    
            from string import printable

            corrected_word = spell.correction(word)
            if word != corrected_word:
                try:
                    vector = model[corrected_word]
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
    preprocess_only = False
    task_description = 'Calculating Word2Vec values'

    model_rel_path = '/model/GoogleNews-vectors-negative300.bin'

    split_ratios = []
    split_ratios_lenght = 0
    output_paths = [None, None]

    anger_dataset = None
    irony_dataset = None

    dfs = [None, None]

    try:
        opts, args = getopt.getopt(argv,'divtsh',['sample_division', 'indent', \
            'validation', 'size=', 'split_ratio=', 'spell_check', 'help', \
            'max_phrase_length=', 'delete-hashtags', 'twitter_model',\
            'preprocess_only', 'anger_dataset=', 'irony_dataset='])
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
        elif o == '--preprocess_only':
            preprocess_only = True
        elif o == '--anger_dataset':
            anger_dataset = a
        elif o == '--irony_dataset':
            irony_dataset = a

    if split_ratios_lenght != 0:
        if validation and split_ratios_lenght != 3:
            print 'Error: 3 split ratios must be provided on validation.'
            sys.exit(2)
        elif divide and not validation and split_ratios_lenght != 2:
            print 'Error: 2 split ratios must be provided on sample division.'
            sys.exit(2)

    if anger_dataset == None or irony_dataset == None:
        print 'Error: Anger and Irony dataset paths must be provided.'
        sys.exit(2)
    else:
        paths = [anger_dataset, irony_dataset]

        for index, path in enumerate(paths):
            if not os.path.isabs(path):
                # Make relative path absolute.
                path = os.path.join(CWD, path)

            if os.path.isfile(path):
                # Get dataset dir path.
                source_path = path.rsplit('/', 1)
                dataset_path = source_path[0]

                # Generate output dir path.
                check_valid_dir(os.path.join(dataset_path, 'json/'))

                output_paths[index] = os.path.join(dataset_path, 'json/' \
                    + source_path[1].rsplit('.csv')[0])

                # Loads CSV into dataframe.
                dfs[index] = pd.read_csv(path, header=0, usecols=LOAD_COLUMNS)

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

    # Twitter preprocessing: replacing URLs and Mentions
    twitter_url_str = (ur'http[s]?://(?:[a-zA-Z]|[0-9]|' \
        ur'[$+*%/@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

    twitter_mention_regex = re.compile(ur'(\s+|^|\.)@\S+')
    twitter_hashtag_regex = re.compile(ur'(\s+|^|\.)#\S+')
    twitter_url_regex = re.compile(twitter_url_str)

    emoticon_str = (ur'(\<[\/\\]?3|[\(\)\\|\*\$][\-\^]?[\:\;\=]' \
            ur'|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)\(\/\|])(?=\s|[\!\.\?]|$)')

    emoticon_regex = re.compile(emoticon_str)

    sentences = []

    outputs = [None, None]
    validation_or_test_outputs = [None, None]
    test_outputs = [None, None]
    distributions = [None, None]

    for index, df in enumerate(dfs):

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
                test_start = (train_length + (df_length - train_length) / 2 ) \
                - 1
        else:
            train_length = round(df_length * split_ratios[0]) -1

            if validation:
                test_start = train_length \
                    + floor(df_length * split_ratios[1]) - 1

        max_word_per_sentence = 0

        preprocess_index = 0

        for tweet in tqdm(df.itertuples(), desc='Tweet preprocessing', \
            total=df_length):

            label = tweet[1]

            # Clean text of new lines.
            preprocessed_tweet = tweet[2].replace('\n', ' ');

            # Unescape possible HTML entities.
            preprocessed_tweet = BeautifulSoup(preprocessed_tweet, \
                'html.parser') .prettify()

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

            sentences.append(tweet_words)

            if divide and preprocess_index >= train_length:
                # Split data into validation file.
                if not validation or preprocess_index <= test_start:
                    validation_or_test_output.append({'label': label, \
                        'words': tweet_words});
                else:
                    # Split data into test validation file.
                    test_output.append({'label': label, 'words': tweet_words});
            elif divide and preprocess_index < train_length:
                # Split data into train file.
                output.append({'label': label, 'words': tweet_words});
            elif not divide:
                # Adds all the data to a single file, without splitting.
                output.append({'label': label, 'words': tweet_words});

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

        outputs[index] = output
        validation_or_test_outputs[index] = validation_or_test_output
        test_outputs[index] = test_output
        distributions[index] = distribution

    # Release memory
    del twitter_url_regex
    del twitter_mention_regex
    del twitter_hashtag_regex
    del emoticon_regex
    del stop_words
    del dfs

    if force_max_phrase_length != None:
        max_word_per_sentence = force_max_phrase_length

    distribution['max_phrase_length'] = max_word_per_sentence
    distribution['model_feature_length'] = NUM_MODEL_FEATURES

    word_id_tokenizer = Tokenizer()
    word_id_tokenizer.fit_on_texts(sentences)
    sequences_phrases = word_id_tokenizer.texts_to_sequences(sentences)
    sentences = pad_sequences(sequences_phrases, maxlen=max_word_per_sentence, \
        padding='post')
    word_index = word_id_tokenizer.word_index

    weights = np.zeros((len(word_index) + 1, NUM_MODEL_FEATURES))
    unknown_words = {}
    # for word, i in tqdm(word_index.items(), desc='Embeddings\' weights ' \
    #     + 'processing', total=len(word_index.items())):

    #     try:
    #         embedding_vector = word2vector(model, word, spell_check)
    #         weights[i] = embedding_vector
    #     except Exception as e:
    #         if word in unknown_words:
    #             unknown_words[word] += 1
    #         else:
    #             unknown_words[word] = 1
    # print "Number of unknown tokens: " + str(len(unknown_words))

    # Release memory.
    del model

    if not preprocess_only:
        for index in xrange(len(outputs)):
            for train_index in trange(len(outputs[index]), \
                desc=task_description, total=len(outputs[index])):

                pass
                # print sequences_phrases[train_index]
                # print outputs[index][train_index]
                # print '-----------------------------------------------'

                # process_sample(model, outputs[index][train_index], \
                #     train_index, max_word_per_sentence, spell_check)

            train_index += 1 

            # Optional if statement to hide console progress bar if not needed.
            if divide:
                for val_test_index in trange(len( \
                    validation_or_test_outputs[index]), \
                    desc=task2_description, \
                    total=len(validation_or_test_outputs[index])):

                    print sequences_phrases[train_index + val_test_index]
                    print validation_or_test_outputs[index][val_test_index]

                    # process_sample(model, validation_or_test_output, \
                    #     val_test_index, max_word_per_sentence, spell_check)

                if validation:
                    for test_index in trange(len(test_outputs[index]), \
                        desc='Calculating Word2Vec test values', \
                        total=len(test_outputs[index])):

                        pass
                        # process_sample(model, test_output, test_index, \
                        #     max_word_per_sentence, spell_check)



    serialization_start_time = datetime.now()
    if not preprocess_only:
        if divide:
            if validation:
                logger.info('Serializing JSON into train, ' \
                    + 'validation and test files.')
                
                # Write validation data to a JSON file.
                serialize_sample(output_path + '_validation.json', \
                    validation_or_test_output, indent)
                
                # Resealse memory.
                del validation_or_test_outputs

            else:
                logger.info('Serializing JSON into train and test files.')

            # Write test data to a JSON file.
            serialize_sample(output_path + '_test.json', \
                test_output, indent)
            
            # Resealse memory.
            del test_outputs
            gc.collect()

            # Write train data to a JSON file.
            serialize_sample(output_path + '_train.json', output, \
                indent)
            
            # Resealse memory.
            del outputs

        else:

            logger.info('Serializing JSON into a file.')

            # Write data to a JSON file.
            serialize_sample(output_path + '.json', \
                output, indent)

            # Resealse memory.
            del outputs

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