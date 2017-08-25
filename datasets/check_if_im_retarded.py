# /usr/bin/env
# -*- coding: utf-8 -*-

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

def main(argv):
	if(len(argv) != 3):
		print 'Error: 3 inputs are required.'
		sys.exit(2)
	else:
		input1 = check_valid_path(argv[0], 'sample 1')
		input2 = check_valid_path(argv[1], 'sample 2')
		input3 = check_valid_path(argv[2], 'sample 3')

		print input1
		print input2
		print input3

		input1 = ujson.load(open(input1, 'r'))
		input2 = ujson.load(open(input2, 'r'))
		input3 = df = pd.read_csv(input3, header=0, \
        dtype={'tweet_id': np.int64})

		if len(input1) != len(input2) != len(input3.index):
			print len(input1)
			print len(input2)
			print len(input3.index)
			print input1[0]
			print input2[0]
			print input3.iloc[0]
			print 'Error: sample have different length, you are retarded'
			sys.exit(2)
		else:
			for index, tweet in enumerate(input1):

				if tweet['label'] != input2[index]['label'] \
					!= input3.iloc[index]['label']:
					
					print tweet['label']
					print input2[index]['index']
					print input3.iloc[index]['label']
					print 'Error: labels do not match, you are retarded.'
					sys.exit(2)

if __name__ == '__main__':
	main(sys.argv[1:])