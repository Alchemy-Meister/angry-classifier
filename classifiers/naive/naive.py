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
    '../../deep-learning/drawing_utils.py'))

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

    manual_class_distribution = {}

    maximum_class = {'name' : '', 'value': 0}

    # Prediction and evaluation for each classifier.
    for index, ordered_matrix_class in enumerate(ordered_matrix_classes):

        manual_class_distribution[ordered_matrix_class] = len(df[ \
            df[COMPULSORY_COLUMNS[3]] == ordered_matrix_class].index)

        if maximum_class['value'] < manual_class_distribution[ \
            ordered_matrix_class ]:

            maximum_class['name'] = ordered_matrix_class
            maximum_class['value'] = manual_class_distribution[ \
                ordered_matrix_class ]

    print manual_class_distribution
    print maximum_class

    # Calculates accuracy per label.
    for label, valid_instance_num in predic_distribution['classes'].iteritems():
        num_predicted_label = float(predicted_distribution['class'][label])

        print label + ' accuracy: ' + str(num_predicted_label \
            / valid_instance_num)

    print ''

    # Calculates accuracy score per classifier.
    for label, num_predicted_label in predicted_distribution['category'] \
        .iteritems():

        print label + ' classifier accuracy: ' + str(num_predicted_label \
            / ( float(predic_distribution['classes'][label] \
            + predic_distribution['classes']['no_' + label]) ))

    print ''

    # Adds the final classification column to the CSV.
    df[result_col] = None

    # Generates final classification according the 2x2 matrix classes.
    for label, condition in matrix_classes.iteritems():
        df.loc[((df[classifiers_name_str[0]] == condition[0]) \
            & (df[classifiers_name_str[1]] == condition[1])) \
            | ((df[classifiers_name_str[0]] == condition[1]) \
            & (df[classifiers_name_str[1]] == condition[0])), result_col] \
            = label

    print predicted_distribution
    del predic_distribution
    print ''

    print 'Dataset length: ' + str(len(df.index))
    print ''

    manual_labeled_tweets = df[~df[COMPULSORY_COLUMNS[3]].isnull()]

    print 'Compared to manual classification'

    y_predict = manual_labeled_tweets[result_col]
    actual_y = manual_labeled_tweets[COMPULSORY_COLUMNS[3]]

    for index, ordered_matrix_class in enumerate(ordered_matrix_classes):
        y_predict = y_predict.replace(to_replace=ordered_matrix_class, \
            value=index)

        actual_y = actual_y.replace(to_replace=ordered_matrix_class, \
            value=index)

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)

    conf_matrix = confusion_matrix(actual_y, y_predict)

    #Y_true, y_predict
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

    predicted_distribution = {}

    # Calculates the accuracy per class.
    for matrix_class in matrix_classes.keys():
        manual_class = manual_labeled_tweets[manual_labeled_tweets[ \
            COMPULSORY_COLUMNS[3] ].str.match(matrix_class)]
        
        manual_class_num = len(manual_class.index)
        predicted_class_num = len(manual_labeled_tweets[manual_labeled_tweets[ \
            result_col ].str.match(matrix_class)].index)

        correct_class_num = len(manual_class[ \
            manual_class[COMPULSORY_COLUMNS[3]] == manual_class[result_col] ] \
            .index)

        predicted_distribution[matrix_class] = manual_class_num

        print( '%s accuracy: %s' % \
            (matrix_class, (float(correct_class_num) / manual_class_num)))

    print predicted_distribution

    # Serializes prediction CSV
    df.to_csv(path_or_buf=os.path.join(dataset_result_output_path, \
        dataset_name + '.csv'), index=False, encoding='utf-8')

if __name__ == '__main__':
    start_time = datetime.now()
    main(sys.argv[1:])
    print "Elapsed time: " + str(datetime.now() - start_time)
