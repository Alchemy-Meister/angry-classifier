# /usr/bin/env
# -*- coding: utf-8 -*-

import codecs
import gensim
import sys
import ujson
import os
import pandas as pd
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()

def check_valid_path(path, desc):
    if not os.path.isabs(path):
        # Make relative path absolute.
        path = os.path.join(CWD, path)

    if not os.path.isfile(path):
        print 'Error: Invalid %s file.' % desc
        sys.exit(2)

    return path

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

def read_file(path):
    f = codecs.open(path, 'r', encoding='utf-8')
    string_file = f.read()
    f.close()
    return string_file

def main(argv):
    if(len(argv) != 3 and len(argv) != 5):
        print 'Error: 3 or 5 inputs are required.'
        sys.exit(2)
    else:
        weights = None
        word_index = None

        model_rel_path = '/../word2vec/model/GoogleNews-vectors-negative300.bin'

        input1 = check_valid_path(argv[0], 'sample 1')
        input2 = check_valid_path(argv[1], 'sample 2')
        input3 = check_valid_path(argv[2], 'sample 3')
        if len(argv) == 5:
            weights = check_valid_path(argv[3], 'weights')
            word_index = check_valid_path(argv[4], 'word_index')

        print input1
        print input2
        print input3
        print weights
        print word_index

        input1 = ujson.loads(read_file(input1), precise_float=True)
        input2 = ujson.loads(read_file(input2))
        input3 = df = pd.read_csv(input3, header=0, \
        dtype={'tweet_id': np.int64})
        if weights != None:
            weights = np.load(weights)
        if word_index != None:
            word_index = ujson.load(open(word_index, 'r'))

        if len(input1) != len(input2) or len(input1) != len(input3.index) \
            or len(input2) != len(input3.index):
            
            print len(input1)
            print len(input2)
            print len(input3.index)
            print input1[0]
            print input2[0]
            print input3.iloc[0]
            print 'Error: sample have different length, you are retarded'
            sys.exit(2)
        else:

            model = gensim.models.Word2Vec.load_word2vec_format( \
            SCRIPT_DIR + model_rel_path, binary=True)

            for index, tweet in enumerate(input1):

                if tweet['label'] != input2[index]['label'] \
                    != input3.iloc[index]['label'] \
                    or input1[index]['id'] != input2[index]['id'] \
                    or input1[index]['id'] != input3.iloc[index]['tweet_id'] \
                    or input2[index]['id'] != input3.iloc[index]['tweet_id'] \
                    or len(tweet['words']) != len(input2[index]['words']):

                    print tweet['label']
                    print input2[index]['label']
                    print input3.iloc[index]['label']
                    print 'Error: labels or IDs do not match, you are retarded.'
                    sys.exit(2)

                for tweet_ndx, word_id in enumerate(input2[index]['words']):

                    word = word_index.keys()[word_index.values().index(word_id)]

                    spell_corrected = False

                    if word != input1[index]['words'][tweet_ndx]:
                        spell_corrected = True

                    try:
                        word_weight = model[word]
                        if not np.allclose(word_weight, \
                            np.array(tweet['word2vec'][tweet_ndx])):

                            print 'Error, word2vec not equal, ' \
                                + 'you are retarded.'

                            if spell_corrected:
                                print 'Alert: different words, check spelling.'
                                print word
                                print input1[index]['words'][tweet_ndx]
                                print input1[index]['words']

                            sys.exit(2)
                    except KeyError:
                        if not np.allclose(np.zeros( \
                            len(tweet['word2vec'][tweet_ndx])),  \
                            np.array(tweet['word2vec'][tweet_ndx])):

                            print 'Error, unknown word is not zero, ' \
                                + 'you are retarded.'

                            if spell_corrected:
                                print 'Alert: different words, check spelling.'
                                print word
                                print input1[index]['words'][tweet_ndx]
                                print input1[index]['words']
                            sys.exit(2)

if __name__ == '__main__':
    main(sys.argv[1:])
