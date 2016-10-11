# /usr/bin/env
# -*- coding: utf-8 -*-

import arff
from datetime import datetime
import HTMLParser
import numpy as np
import pandas as pd
import re
import string
import sys, getopt

dataset_path = './../datasets/crowdflower/'
dataset_filename = 'text_emotion.csv'
arff_folder = 'arff/'
binary_tag = 'binary_'
output_filename = 'crowdflower.arff'

def main(argv):
    # Flag to performe binary classification.
    binary_classification = False

    # Define and initialize arff data dictionary.
    data_dict = {'attributes': [], 'data': [], 'description': '', \
        'relation': ''}
    data_dict['relation'] = 'Twitter-Messages'

    # Checks if parameter is being used.
    try:
        opts, args = getopt.getopt(argv,'b',['binary_classification'])
    except getopt.GetoptError:
        print 'ERROR: Unknown parameter. Usage: crowdflower2arff.py [-b]'
        sys.exit(2)

    for o, a in opts:
        if o == '-b' or o == '--binary_classification':
            binary_classification = True

    # Loads CSV into dataframe.
    df = pd.read_csv(dataset_path + dataset_filename, usecols=[1,3], header=0)

    # Loads dictionary classes depending on if binary classification flag is
    # enabled.
    if binary_classification:
        data_dict['attributes'] = [('text', 'STRING'), ('@@class@@', ['anger', \
            'no_anger'])]

        # Creates a binary sub-samples of the original dataframe
        anger_tweets = df[ (df.sentiment == 'anger') \
            | (df.sentiment == 'hate') ].copy()
        no_anger_tweets = df[ (df.sentiment != 'anger') \
            | (df.sentiment != 'hate')].copy()

        # Replace sentiment labels for binary classification
        anger_tweets['sentiment'] = 'anger'
        no_anger_tweets['sentiment'] = 'no_anger'

        # Select a random subset of len(anger_tweets) without replacement.
        no_anger_tweets = no_anger_tweets.take( \
            np.random.permutation(len(no_anger_tweets))[:len(anger_tweets)] )

        # Merge both dataframes.
        anger_tweets = anger_tweets.append(no_anger_tweets, ignore_index=True)

        # Shuffle
        df = anger_tweets.sample(frac=1).reset_index(drop=True)

    else:
        data_dict['attributes'] = [('text', 'STRING'), ('@@class@@', ['anger', \
            'boredom', 'enthusiasm', 'empty', 'fun', 'happiness', 'hate', \
            'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry'])]

    for tweet in df.itertuples():
        # Twitter preprocessing: replacing URLs and Mentions
        twitter_url_str = (ur'http[s]?://(?:[a-zA-Z]|[0-9]|' \
            ur'[$+*%/@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

        twitter_mention_regex = re.compile(ur'(\s+|^|\.)@\S+')
        twitter_hashtag_regex = re.compile(ur'(\s+|^|\.)#\S+')
        twitter_url_regex = re.compile(twitter_url_str)

        # Clean text of new lines.
        preprocessed_tweet = tweet[2].replace('\n', ' ');

        # Unscape possible HTML entities.
        preprocessed_tweet = HTMLParser.HTMLParser() \
            .unescape(preprocessed_tweet)

        # Remove urls and mentions with representative key code.
        preprocessed_tweet = re.sub(twitter_url_regex, 'URL', \
            preprocessed_tweet)
        preprocessed_tweet = re.sub(twitter_mention_regex, ' MENTION', \
            preprocessed_tweet)

        # Removes punctiation including hashtags.
        preprocessed_tweet =  preprocessed_tweet.encode('utf-8') \
            .translate(None, string.punctuation)
        preprocessed_tweet.decode('utf-8')

        # Trims generated string.
        preprocessed_tweet = preprocessed_tweet.strip()

        # Escape \ character.
        preprocessed_tweet = preprocessed_tweet.replace('\\', '\\\\')

        data_dict['data'].append([preprocessed_tweet, tweet[1]])


    # Generate arff format string.
    arff_data = arff.dumps(data_dict)

    # Write arff string into file.
    if binary_classification:
        arff_output = open(dataset_path + arff_folder + binary_tag \
            + output_filename, 'w+')
    else:
        arff_output = open(dataset_path + arff_folder + output_filename, 'w+')
    arff_output.write(arff_data)
    arff_output.close()

if __name__ == "__main__":
    start_time = datetime.now()
    main(sys.argv[1:])
    print "Elapsed time: " + str(datetime.now() - start_time)