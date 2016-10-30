# /usr/bin/env
# -*- coding: utf-8 -*-

import codecs
from datetime import datetime
import gensim
import HTMLParser
import json
import logging
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import string
from tqdm import tqdm
import ujson

start_time = datetime.now()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

max_word_per_sentence = 0

dataset_path = './../datasets/crowdflower/'
dataset_filename = 'text_emotion.csv'
output_filename = 'json/crowdflower.json'

# Loads NLTK's stopwords for English.
stop_words = stopwords.words("english")

# Loads Word2Vec model.
# Load Google's pre-trained Word2Vec model.
model = gensim.models.Word2Vec.load_word2vec_format( \
    './../word2vec/model/GoogleNews-vectors-negative300.bin', binary=True)

logger.info('Model loading complete, elapsed time: %s', \
    str(datetime.now() - start_time))

# Loads CSV into dataframe.
df = pd.read_csv(dataset_path + dataset_filename, usecols=[1,3], header=0)

output = []

df_length = len(df.index)

for tweet in tqdm(df.itertuples(), desc='Tweet preprocessing', total=df_length):

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

    output.append({'label': tweet[1], 'words': tweet_words, 'word2vec': []});

    # Update largest sentence's word number.
    if max_word_per_sentence < word_count:
        max_word_per_sentence = word_count

index = 0
for tweet in tqdm(df.itertuples(), desc="Calculating Word2Vec values", \
        total=df_length):
    
    word2vec = []
    for word in output[index]['words']:
        try:
            output[index]['word2vec'].append(model[word].tolist())
        except KeyError:
            # Check the spelling in a dictionary.
            output[index]['word2vec'].append(np.zeros(300).tolist())

    # Remove word list from the dict
    output[index].pop('words', None)

    # Update loop counter.
    index += 1

logger.info('Serializing JSON into a File.')

serialization_start_time = datetime.now()

# Write data to a JSON file.
ujson.dump(output, codecs.open(dataset_path + output_filename, 'w', \
    encoding='utf-8'))

logger.info('Serialization finished, elapsed time: %s', \
    str(datetime.now() - serialization_start_time))

logger.info('Total elapsed time: %s', \
    str(datetime.now() - start_time))