# /usr/bin/env
# -*- coding: utf-8 -*-

import csv
import getopt
import os
from os import listdir
import sys
import ujson

csv_experiment_order = ['clean_embeddings', 'trainable_embeddings', \
    'trainable_embeddings_single_dense', 'trainable_embeddings_double_dense', \
    'trainable_embeddings_batch_conv_activ', \
    'trainable_embeddings_double_dense_with_0.4_drop', \
    'trainable_embeddings_double_dense_with_0.8_drop', \
    'trainable_embeddings_double_dense_with_0.4_drop_conv_actv_batch', \
    'trainable_embeddings_double_dense_with_0.8_drop_conv_actv_batch', \
    'trainable_embeddings_double_dense_with_0.4_drop_relu_softmax_batch', \
    'trainable_embeddings_double_dense_with_0.8_drop_relu_softmax_batch', \
    'trainable_embeddings_double_dense_with_0.4_drop_both_batch', \
    'trainable_embeddings_double_dense_with_0.8_drop_both_batch', \
    'trainable_embeddings_single_dense_with_0.4_drop_conv_actv_batch', \
    'trainable_embeddings_512_filters_double_dense_with_0.4_drop_conv_actv_batch']

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CWD = os.getcwd()

def check_valid_path(path, desc):
    if not os.path.isabs(path):
        # Make relative path absolute.
        path = os.path.join(CWD, path)

    if not os.path.isfile(path):
        print 'Error: Invalid %s file.' % desc
        print path
        sys.exit(2)

    return path

def to_absolute_dir(dir_name):
    if not os.path.isabs(dir_name):
        dir_name = os.path.join(CWD, dir_name)

        if not os.path.exists(dir_name):
            print 'Error: %s doesn\'t exist' % dir_name
            sys.exit(1)
    return dir_name

def main(argv):
    dataset_name = None
    subexperiments = None
    use_avg = False

    try:
        opts, args = getopt.getopt(argv,'', ['dataset=', 'subexperiments=', \
            'avg'])
    except getopt.GetoptError:
        print 'Error: Unknown parameter.' 
        sys.exit(2)

    for o, a in opts:
        if o == '--dataset':
            dataset_name = to_absolute_dir(a)
        elif o == '--subexperiments':
            subexperiments = a.split(',')
        elif o == '--avg':
            use_avg = True

    results_path = os.path.join(dataset_name, 'results/embeddings/second_phase/')

    experiments_list = listdir(results_path)

    with open('some_results.csv', 'w') as f:
        writer = csv.writer(f)

        for experiment in csv_experiment_order:
            experiment_folder = os.path.join(results_path, \
                    experiment)

            for subexperiment in subexperiments:
                subexperiment_folder = os.path.join(experiment_folder, \
                    subexperiment + '/best/')

                if use_avg:
                    best_result = os.path.join(subexperiment_folder, \
                        'avg_result.json')
                else:
                    for file in listdir(subexperiment_folder):
                        if (not file.endswith('.json')) and \
                        (not file.endswith('.png')):

                            best_result = os.path.join(subexperiment_folder, \
                                file)

                best_result = ujson.load(open(best_result, 'r'))

                writer.writerow([best_result['acc'], \
                    best_result['f1_macro'], best_result['recall_macro'], \
                    best_result['precision_macro']])

                print '%s - %s %s' % (experiment, subexperiment, \
                    best_result['f1_macro'])


if __name__ == '__main__':
    main(sys.argv[1:])