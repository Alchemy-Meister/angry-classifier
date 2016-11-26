# /usr/bin/env
# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
from sklearn import svm
import ujson

DATASET_PATH = './../datasets/crowdflower/json/crowdflower'

def main():
    class_distribution = ujson.load(open(DATASET_PATH + '_distribution.json', \
        'r'))
    labels = [key for key in class_distribution['classes'].keys()]
    num_categories = len(class_distribution['classes'])
    max_phrase_length = class_distribution['max_phrase_length']
    model_size = class_distribution['model_feature_length']

    y = np.array(labels, dtype='str')

    print y


if __name__ == '__main__':
    start_time = datetime.now()
    main()
    print "Elapsed time: " + str(datetime.now() - start_time)