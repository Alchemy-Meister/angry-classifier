# /usr/bin/env
# -*- coding: utf-8 -*-

from datetime import datetime
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Merge
from keras.models import Sequential
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical, probas_to_classes
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

# Hack to import from parent folder.
sys.path.append(SCRIPT_DIR + '/..')
from drawing.drawing_utils import EpochDrawer, ConfusionMatrixDrawer

USAGE_STRING = 'Usage: repressed-anger.py [-h] [--help] ' \
    + '[--dataset=path_to_original_dataset]' \
    + '[--anger_dir=path_to_anger_dir] ' \
    + '[--anger_model_weights=anger_weights_filename] ' \
    + '[--anger_distribution=anger_distribution_path]' \
    + '[--irony_dir=path_to_irony_dir] ' \
    + '[--irony_model_weights=anger_weights_filename] ' \
    + '[--irony-distribution=irony_distribution_path]'

CSV_COLUMNS = ['tweet_id', 'label', 'author', 'content', 'manual_label']

# Author column is not required.
COMPULSORY_COLUMNS = list(CSV_COLUMNS)
del COMPULSORY_COLUMNS[2]

def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

def prepare_samples(piece_path, max_phrase_length, model_length):
    X = []
    with open(piece_path, 'r') as piece:
        program = json.load(piece, object_hook=json_numpy_obj_hook)
       
        for phrase in program:
            X.append(np.array(phrase['words']))
    X = np.array(X)
    X = pad_sequences(X, maxlen=max_phrase_length, padding='post')
    return X

def one_hot_encoder(total_classes):
    encoder = {}
    array_size = len(total_classes)
    for i, code in enumerate(total_classes):
        vector = np.zeros(array_size)
        vector[i] = 1.0
        encoder[code] = vector.tolist()
    return encoder

def load_model(model_path, weights_path):
    model_file = open(model_path, 'r')
    model_str = model_file.read()
    model_file.close()

    model = model_from_json(model_str)
    model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', \
        metrics=['accuracy', 'mse', 'mae'])

    return model

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

    result_col = 'classification'
    classifiers_name_str = ['anger', 'irony']
    classifiers_attr_str = ['dir', 'model', 'weights', 'distribution']
    
    matrix_classes = {
        'explicit anger': ['anger', 'no_irony'],
        'repressed anger': ['anger', 'irony'],
        'normal': ['no_anger', 'no_irony'],
        'irony': ['no_anger', 'irony']
        }

    ordered_matrix_classes = OrderedDict(matrix_classes)
    ordered_matrix_classes = ordered_matrix_classes.keys()

    classifiers = dict()
    for classifier_name in classifiers_name_str:
        classifiers[classifier_name] = dict.fromkeys(classifiers_attr_str, None)

    classifiers_length = len(classifiers.keys())
    max_phrase_length = None
    model_length = None
    predic_distribution = None

    try:
        opts, args = getopt.getopt(argv,'h',['dataset=', 'anger_dir=', \
            'anger_weights_filename=', 'anger_distribution=','irony_dir=', \
            'irony_weights_filename=', 'irony_distribution=', 'help'])
    except getopt.GetoptError:
        print 'Error: Unknown parameter. %s' % USAGE_STRING
        sys.exit(2)

    for o, a in opts:
        if o == '-h' or o == '--help':
            print USAGE_STRING
            sys.exit(0)
        if o == '--dataset':
            dataset_path = check_valid_path(a, 'dataset')
            dataset_split_path = dataset_path.rsplit('/', 1)
            
            dataset_name = dataset_split_path[1].split('.csv')[0]

            word2vec_dataset_path = check_valid_path(dataset_split_path[0] \
                + '/json/' + dataset_name + '.json', \
                'word2vec dataset')

            # Loads prediction dataset distribution file.
            predic_distribution = ujson.load( open(check_valid_path( \
                word2vec_dataset_path.split('.json')[0] \
                + '_distribution.json', 'predict distribution'), 'r') )

            dataset_result_output_path = os.path.join(SCRIPT_DIR, \
                dataset_name)
            check_valid_dir(dataset_result_output_path)

            results_output_path = os.path.join(dataset_result_output_path, \
                'results')

            check_valid_dir(results_output_path)

        elif o == '--anger_dir':
            class_dir = check_valid_dir(a)
            classifiers[classifiers_name_str[0]][classifiers_attr_str[0]] \
                = class_dir
            classifiers[classifiers_name_str[0]][classifiers_attr_str[1]] \
                = os.path.join(class_dir, 'model/model.json')
        elif o == '--anger_weights_filename':
            classifiers[classifiers_name_str[0]][classifiers_attr_str[2]] \
                = a
        elif o == '--anger_distribution':
            classifiers[classifiers_name_str[0]][classifiers_attr_str[3]] \
                = check_valid_path(a, 'anger distribution')
        elif o == '--irony_dir':
            class_dir = check_valid_dir(a)
            classifiers[classifiers_name_str[1]][classifiers_attr_str[0]] \
                = class_dir
            classifiers[classifiers_name_str[1]][classifiers_attr_str[1]] \
                = os.path.join(class_dir, 'model/model.json')
        elif o == '--irony_weights_filename':
            classifiers[classifiers_name_str[1]][classifiers_attr_str[2]] \
                = a
        elif o == '--irony_distribution':
            classifiers[classifiers_name_str[1]][classifiers_attr_str[3]] \
                = check_valid_path(a, 'irony distribution')

    # Input error detection.
    if dataset_path == None:
        print 'Error: dataset path is required.\n %s' % USAGE_STRING
        sys.exit(2)

    elif classifiers[classifiers_name_str[0]][classifiers_attr_str[0]] == None \
        or classifiers[classifiers_name_str[1]][classifiers_attr_str[0]] \
        == None:

        print 'Error: anger and irony directories are required.\n %s' \
            % USAGE_STRING
        sys.exit(2)
    elif classifiers[classifiers_name_str[0]][classifiers_attr_str[2]] == None \
        or classifiers[classifiers_name_str[1]][classifiers_attr_str[2]] \
        == None:
        
        print 'Error: anger and irony model weights filename are required.\n' \
            % USAGE_STRING
        sys.exit(2)

    elif classifiers[classifiers_name_str[0]][classifiers_attr_str[3]] == None \
        or classifiers[classifiers_name_str[1]][classifiers_attr_str[3]] \
        == None or predic_distribution == None:
        
        print 'Error: anger, irony predict distribution paths are required.\n' \
            % USAGE_STRING
        sys.exit(2)
    else:
        for classifier_name, classifier_dict in classifiers.iteritems():
            classifier_dict[classifiers_attr_str[3]] = \
                ujson.load(open(classifier_dict[classifiers_attr_str[3]], 'r'))

            if max_phrase_length is None:
                max_phrase_length = classifier_dict[classifiers_attr_str[3]] \
                    ['max_phrase_length']
            elif max_phrase_length != classifier_dict[ \
                classifiers_attr_str[3] ]['max_phrase_length']:

                print 'Error: both classifiers must be pre-trained with ' \
                + 'same max_phrase_length.'
                sys.exit(2)

            if model_length == None:
                model_length = classifier_dict[ \
                    classifiers_attr_str[3] ]['model_feature_length']
            elif model_length != classifier_dict[ \
                classifiers_attr_str[3] ]['model_feature_length']:

                print 'Error: both classifiers much be pre-trained with ' \
                    + 'same model feature length.'
                sys.exit(2)

        if max_phrase_length != predic_distribution['max_phrase_length']:
            print 'Error: prediction and pre-trained word embeddings must ' \
                + 'have same max_phrase_length.'
            sys.exit(2)
        elif model_length != predic_distribution['model_feature_length']:
            print 'Error: prediction and pre-trained word embeddings must ' \
                + 'have same model_feature_length.'
            sys.exit(2)

    # Load original dataset.
    df = pd.read_csv(dataset_path, header=0, \
        dtype={COMPULSORY_COLUMNS[0]: np.int64})

    # Loads word embedding dataset.
    X_predict = prepare_samples(word2vec_dataset_path, max_phrase_length, \
    model_length)

    predicted_distribution = {'class': {}, 'category': {}}

    # Prediction and evaluation for each classifier.
    for classifier_name, classifier_dict in classifiers.iteritems():

        # Load model weights path into the dictionary.
        classifier_dict[classifiers_attr_str[2]] = os.path.join( \
            classifier_dict[classifiers_attr_str[0]], 'model_weights/' \
            + classifier_dict[classifiers_attr_str[2]])

        model = load_model(classifier_dict[classifiers_attr_str[1]], \
            classifier_dict[classifiers_attr_str[2]])

        # Dataset prediction.
        y_predict = model.predict(X_predict)
        # Calculates prediction's one hot encoder
        y_predict = probas_to_classes(y_predict)

        # Adds one hot encoder into the original dataset as a new column.
        df[classifier_name] = y_predict

        # Get the labels available for the current classifier.
        labels = classifier_dict[classifiers_attr_str[3]]['classes'].keys()

        for index in xrange(len(labels)):
            # Changes the one hot encoder values into representing class names.
            df.loc[df[classifier_name] == index, classifier_name] \
                = labels[index]

            # Saves the number of correctly classified tweets number per label.
            predicted_distribution['class'][labels[index]] = len(df[ 
                (df[COMPULSORY_COLUMNS[1]] == labels[index]) \
                & (df[classifier_name] == df[COMPULSORY_COLUMNS[1]]) ].index)

        # Saves the number of correctly classified tweets per classifier.
        predicted_distribution['category'][classifier_name] = \
            len(df[df[classifier_name] == df[COMPULSORY_COLUMNS[1]]].index)

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
