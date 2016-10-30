# /usr/bin/env
# -*- coding: utf-8 -*-

from datetime import datetime
import pandas as pd
import sys
from tqdm import tqdm
import tweepy
import unicodewriter as csv

DATASET_PATH = './../datasets/'
FILE_PATH = ''

CONSUMER_KEY = 'PzM5sZEdAvJdFvMTPWZrpa0sm'
CONSUMER_SECRET = 'NVZaUabS2Ul50NSNQ6YOx8Wkm4qqlbfVgf7larq6ia6h9sBWCQ'
ACCESS_TOKEN = '425711895-jyJTMmTQ5YssL07h8BrEye9M5q4nIgToosHD0JI9'
ACCESS_TOKEN_SECRET = 'OHHUO1D87r89K1XmfZOirrBvsYMUWCZb8eV4nsNUrzCKp'

def main(argv):

    if len(argv) != 1:
        # Checks if path to file is introduced.
        print 'ERROR: Missing argument. Usage: tweet-downloader.py' \
            ' path-to-tweet-ids.txt'
        sys.exit(2)

    FILE_PATH = argv[0]

    # OAuth authentication
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    
    # Tweepy API declaration.
    api = tweepy.API(auth)

    tweets = []

    try:
        df = pd.read_csv(DATASET_PATH + FILE_PATH,sep='\t', header=None)

        for row in df.itertuples():
            try:
                tweet = api.get_status(row._1)
                tqdm.pandas(desc="my bar!")
                tweets.add(tweet.text)
                
            except:
                pass
    except:
        print DATASET_PATH + FILE_PATH + ' does not exist.'

    csv_header = ['tweet_id', 'sentiment', 'author', 'content']

    csv_writer = csv(DATASET_PATH + FILE_PATH, bw)
    csv_writer.writerow(csv_header)



if __name__ == "__main__":
    start_time = datetime.now()
    main(sys.argv[1:])
    print "Elapsed time: " + str(datetime.now() - start_time)