# /usr/bin/env
# -*- coding: utf-8 -*-

import codecs
from datetime import datetime
import logging
import pandas as pd
import sys
from tqdm import tqdm
import tweepy
from level_filter import LevelFilter
import ujson

DATASET_PATH = './../datasets/'
JSON_PATH = '/json/'
FILE_PATH = ''
OUTPUT_FILENAME = ''

CONSUMER_KEY = 'PzM5sZEdAvJdFvMTPWZrpa0sm'
CONSUMER_SECRET = 'NVZaUabS2Ul50NSNQ6YOx8Wkm4qqlbfVgf7larq6ia6h9sBWCQ'
ACCESS_TOKEN = '425711895-jyJTMmTQ5YssL07h8BrEye9M5q4nIgToosHD0JI9'
ACCESS_TOKEN_SECRET = 'OHHUO1D87r89K1XmfZOirrBvsYMUWCZb8eV4nsNUrzCKp'

logger = logging.getLogger('tweet-downloader')
logger.setLevel(logging.INFO)

console_logger = logging.StreamHandler()
console_logger.setLevel(logging.INFO)
console_logger.addFilter(LevelFilter(logging.INFO))

logger.addHandler(console_logger)

error_codes = {}

def check_file_name(argv):
    if len(argv) != 1:
        # Checks if path to file is introduced.
        logger.info('ERROR: Missing argument. Usage: tweet-downloader.py' \
            ' path-to-tweet-ids.txt')
        sys.exit(2)

def no_code_error(message, tweet_id):
    if 'no-code' not in error_codes:
        error_codes['no-code'] = {} 
                    
    if message not in error_codes['no-code']:
        error_codes['no-code'][message] = []
                    
    error_codes['no-code'][message].append(tweet_id)

def dict_error(message, tweet_id):
    code = message['code']
    if code not in error_codes:
        message = message['message']
        error_codes[code] = {}
        error_codes[code]['message'] = message
        error_codes[code]['tweets'] = []

    error_codes[code]['tweets'].append(tweet_id)

def main(argv):

    check_file_name(argv)

    FILE_PATH = str(argv[0]).strip()
    OUTPUT_FILENAME = FILE_PATH.split('.')[0]

    split_path = OUTPUT_FILENAME.split('/')

    # OAuth authentication
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    
    # Tweepy API declaration.
    api = tweepy.API(auth, wait_on_rate_limit=True)

    tweets = []

    df = pd.read_csv(DATASET_PATH + FILE_PATH,sep='\t', header=None, \
        dtype={'tweet_id': int})
    
    # Add two more columns to the dataframe.
    df[2] = ''
    df[3] = ''

    df_length = len(df.index)

    # save tweet_id as integer instead of float.
    df[0] = df[0].astype(int)

    index = 0
    for row in tqdm(df.itertuples(), desc='Downloading Tweets', \
        total=df_length):

        try:
            tweet = api.get_status(row._1)
            
            # Add author and tweet text to the dataframe.
            df.iloc[index, df.columns.get_loc(2)] = tweet.author.screen_name
            df.iloc[index, df.columns.get_loc(3)] = tweet.text

        except Exception, e:
            
            # Generates a JSON file about unavailable tweets.
            message = e.message
            if isinstance(message, list):
                message = message[0]
                
                if isinstance(message, dict):
                    dict_error(message, row._1)

                else:
                    no_code_error(message, row._1)

            elif isinstance(message, dict):
                print message
                print message.keys()
                try:
                    dict_error(message, row._1)
                except:
                    pass
            else:
                no_code_error(message, row._1)

        index += 1

    # Remove rows with unavailable tweets.
    df = df.drop(df[df[3] == ''].index)

    # save tweet_id as integer instead of float.
    df[0] = df[0].astype(int)

    logger.info('Serializing Dataframe into a CSV File.')
    serialization_start_time = datetime.now()

    df.to_csv(path_or_buf=DATASET_PATH + OUTPUT_FILENAME + '.csv', \
        header=['tweet_id', 'sentiment', 'author', 'content'], index=False,
        encoding='utf-8')

    split_path = OUTPUT_FILENAME.split('/')

    with codecs.open(DATASET_PATH + split_path[0] + JSON_PATH + split_path[1] \
        + '_error.json', 'w', encoding='utf-8') as dout:
        
        dout.write(ujson.dumps(error_codes, indent=4))

    logger.info('Serialization finished, elapsed time: %s', \
        str(datetime.now() - serialization_start_time))

if __name__ == "__main__":
    start_time = datetime.now()
    main(sys.argv[1:])
    print "Elapsed time: " + str(datetime.now() - start_time)
