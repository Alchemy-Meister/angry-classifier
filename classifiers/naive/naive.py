# /usr/bin/env
# -*- coding: utf-8 -*-

from datetime import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    precision_score, recall_score

import codecs
import time, os, gc, sys
import getopt
import ujson, json, base64
import numpy as np
import pandas as pd
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()

import imp
drawing_utils = imp.load_source('drawing_utils', os.path.join(SCRIPT_DIR, \
    '../../deep-learning/drawing/drawing_utils.py'))

from drawing_utils import EpochDrawer, ConfusionMatrixDrawer

USAGE_STRING = 'Usage: naive.py [-h] [--help] path_to_original_dataset'

CSV_COLUMNS = ['tweet_id', 'label', 'author', 'content', 'manual_label']

# Author column is not required.
COMPULSORY_COLUMNS = list(CSV_COLUMNS)
del COMPULSORY_COLUMNS[2]

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
    dataset_path = None
    word2vec_dataset_path = None

    dataset_name = None
    dataset_result_output_path = None
    results_output_path = None

    matrix_classes = {
        'explicit anger': ['anger', 'no_irony'],
        'repressed anger': ['anger', 'irony'],
        'normal': ['no_anger', 'no_irony'],
        'irony': ['no_anger', 'irony']
        }

    ordered_matrix_classes = OrderedDict(matrix_classes)
    ordered_matrix_classes = ordered_matrix_classes.keys()

    try:
        opts, args = getopt.getopt(argv,'h',['help'])
    except getopt.GetoptError:
        print 'Error: Unknown parameter. %s' % USAGE_STRING
        sys.exit(2)

    for o, a in opts:
        if o == '-h' or o == '--help':
            print USAGE_STRING
            sys.exit(0)

    if len(args) != 1:
        print 'Error: dataset path is required. %s' % USAGE_STRING
        sys.exit(2)
    else:

        dataset_path = check_valid_path(args[0], 'dataset')
        dataset_split_path = dataset_path.rsplit('/', 1)

        dataset_name = dataset_split_path[1].split('.csv')[0]

        dataset_result_output_path = os.path.join(SCRIPT_DIR, \
            dataset_name)
        check_valid_dir(dataset_result_output_path)

        results_output_path = os.path.join(dataset_result_output_path, \
            'results')

        check_valid_dir(results_output_path)

    # Load original dataset.
    df = pd.read_csv(dataset_path, header=0, \
        dtype={COMPULSORY_COLUMNS[0]: np.int64})

    df = df[~df[COMPULSORY_COLUMNS[3]].isnull()]
    df_length = len(df.index)

    manual_class_distribution = {}
    maximum_class = {'name' : '', 'value': 0}

    actual_y = df[COMPULSORY_COLUMNS[3]]

    for index, ordered_matrix_class in enumerate(ordered_matrix_classes):

        manual_class_distribution[ordered_matrix_class] = len(df[ \
            df[COMPULSORY_COLUMNS[3]] == ordered_matrix_class].index)

        if maximum_class['value'] < manual_class_distribution[ \
            ordered_matrix_class ]:

            maximum_class['name'] = ordered_matrix_class
            maximum_class['value'] = manual_class_distribution[ \
                ordered_matrix_class ]

        actual_y = actual_y.replace(to_replace=ordered_matrix_class, \
            value=index)

    y_predict = pd.Series([maximum_class['name'] for x in xrange(df_length)])

    for index, ordered_matrix_class in enumerate(ordered_matrix_classes):

        y_predict = y_predict.replace(to_replace=ordered_matrix_class, \
            value=index)

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)

    conf_matrix = confusion_matrix(actual_y, y_predict)

    ConfusionMatrixDrawer(conf_matrix, classes=ordered_matrix_classes, \
        str_id=timestamp, title='Confusion matrix, without normalization', \
        folder=results_output_path)
    ConfusionMatrixDrawer(conf_matrix, classes=ordered_matrix_classes, \
        normalize=True, title='Normalized confusion matrix', \
        folder=results_output_path, str_id=timestamp)

    acc = accuracy_score(actual_y, y_predict)
    print "The accuracy of the model using scikit is: " + str(acc)

    f1_macro = f1_score(actual_y, y_predict, average="macro")
    recall_macro = recall_score(actual_y, y_predict, average="macro")
    precision_macro = precision_score(actual_y, y_predict, average="macro")
    f1_micro = f1_score(actual_y, y_predict, average="micro")
    recall_micro = recall_score(actual_y, y_predict, average="micro")
    precision_micro = precision_score(actual_y, y_predict, average="micro")

    results = {}
    results['precision_macro'] = precision_macro
    results['recall_macro'] = recall_macro
    results['f1_macro'] = f1_macro
    results['precision_micro'] = precision_micro
    results['recall_micro'] = recall_micro
    results['f1_micro'] = f1_micro
    results['acc'] = acc

    with codecs.open(results_output_path + '/' + timestamp, 'w', \
        encoding='utf-8') as file:

        file.write(ujson.dumps(results, indent=4))

if __name__ == '__main__':
    start_time = datetime.now()
    main(sys.argv[1:])
    print "Elapsed time: " + str(datetime.now() - start_time)
