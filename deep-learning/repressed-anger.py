# /usr/bin/env
# -*- coding: utf-8 -*-

from datetime import datetime
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Merge
from keras.models import Sequential
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical, probas_to_classes
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    precision_score, recall_score

import codecs
import time, os, gc, sys
import getopt
import ujson, json, base64
import numpy as np
import pandas as pd
from drawing_utils import EpochDrawer, ConfusionMatrixDrawer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()

TARGETS = ['train', 'test', 'all']

USAGE_STRING = 'Usage: cnn.py [-h] [--help] ' \
    + '[--dataset=path_to_original_dataset]' \
    + '[--anger_dir=path_to_anger_dir] ' \
    + '[--anger_model_weights=anger_weights_filename] ' \
    + '[--anger_distribution=anger_distribution_path]' \
    + '[--irony_dir=path_to_irony_dir] ' \
    + '[--irony_model_weights=anger_weights_filename] ' \
    + '[--irony-distribution=irony_distribution_path]'

CSV_COLUMNS = ['tweet_id', 'label', 'author', 'content']

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

def prepare_samples(piece_path, max_phrase_length):
    X = []
    with open(piece_path, 'r') as piece:
        program = json.load(piece, object_hook=json_numpy_obj_hook)

        for phrase in program:
            X.append(np.array(phrase['word2vec']))
    X = np.array(X)

    if K.image_dim_ordering() == 'th':
        X = X.reshape(X.shape[0], 1, max_phrase_length, 300)
        #input_shape = (1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], max_phrase_length, 300, 1)
        #input_shape = (img_rows, img_cols, 1)
    return X

def one_hot_encoder(total_classes):
    encoder = {}
    array_size = len(total_classes)
    for i, code in enumerate(total_classes):
        vector = np.zeros(array_size)
        vector[i] = 1.0
        encoder[code] = vector.tolist()
    return encoder

def find_1(one_hot):
    for i, e in enumerate(one_hot):
        if e == 1:
            return i
    print 'not found'

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

def test(test_path, model, labels, max_phrase_length, output_path):
    # Using keras evaluation method,
    # just check if I am doing it right with SciKit.
    X_test, y_test = prepare_samples(test_path, labels, max_phrase_length)

    y_predict = model.evaluate([X_test, X_test, X_test], y_test, verbose=0, \
        batch_size=BATCH_SIZE)

    print 'The metrics keras is evaluating are: ' + str(model.metrics_names) \
    + ' and its results: ' + str(y_predict)
    # Using Scikit in order to evaluate the model.
    y_predict = model.predict([X_test, X_test, X_test])
    y_predict = probas_to_classes(y_predict)
    y_raw_test = []
    for y1 in y_test:
        y_raw_test.append(find_1(y1))

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)

    #Y_true, y_predict
    conf_matrix = confusion_matrix(y_raw_test, y_predict)
    ConfusionMatrixDrawer(conf_matrix, classes=labels, str_id=timestamp, \
        title='Confusion matrix, without normalization', folder=output_path)
    ConfusionMatrixDrawer(conf_matrix, classes=labels, normalize=True, \
        title='Normalized confusion matrix', folder=output_path, \
        str_id=timestamp)

    print conf_matrix
    
    acc = accuracy_score(y_raw_test, y_predict)
    print "The accuracy of the model using scikit is: " + str(acc)

    f1_macro = f1_score(y_raw_test, y_predict, average="macro")
    recall_macro = recall_score(y_raw_test, y_predict, average="macro")
    precision_macro = precision_score(y_raw_test, y_predict, average="macro")
    f1_micro = f1_score(y_raw_test, y_predict, average="micro")
    recall_micro = recall_score(y_raw_test, y_predict, average="micro")
    precision_micro = precision_score(y_raw_test, y_predict, average="micro")

    results = {}
    results['precision_macro'] = precision_macro
    results['recall_macro'] = recall_macro
    results['f1_macro'] = f1_macro
    results['precision_micro'] = precision_micro
    results['recall_micro'] = recall_micro
    results['f1_micro'] = f1_micro
    results['acc'] = acc

    with codecs.open(output_path + '/' + timestamp, 'w', \
        encoding='utf-8') as file:

        file.write(ujson.dumps(results, indent=4))

def main(argv):
    dataset_path = None
    word2vec_dataset_path = None

    result_col = 'classification'
    classifiers_name_str = ['anger', 'irony']
    classifiers_attr_str = ['dir', 'model', 'weights', 'distribution']
    
    matrix_classes = {
        'explicit anger': ['anger', 'no_irony'],
        'repressed anger': ['anger', 'irony'],
        'normal': ['no_anger', 'no_irony'],
        'irony': ['no_anger', 'irony']
        }

    classifiers = dict()
    for classifier_name in classifiers_name_str:
        classifiers[classifier_name] = dict.fromkeys(classifiers_attr_str, None)

    classifiers_length = len(classifiers.keys())
    max_phrase_length = None
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
            dataset_split_path = dataset_path.rsplit('/' ,1)
            word2vec_dataset_path = check_valid_path(dataset_split_path[0] \
                + '/json/' + dataset_split_path[1].split('.csv')[0] + '.json', \
                'word2vec dataset')
            predic_distribution = ujson.load( open(check_valid_path( \
                word2vec_dataset_path.split('.json')[0] \
                + '_distribution.json', 'predict distribution'), 'r') )

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

    if dataset_path == None:
        print 'Error, dataset path is required.\n %s' % USAGE_STRING
        sys.exit(2)

    elif classifiers[classifiers_name_str[0]][classifiers_attr_str[0]] == None \
        or classifiers[classifiers_name_str[1]][classifiers_attr_str[0]] \
        == None:

        print 'Error, anger and irony directories are required.\n %s' \
            % USAGE_STRING
        sys.exit(2)
    elif classifiers[classifiers_name_str[0]][classifiers_attr_str[2]] == None \
        or classifiers[classifiers_name_str[1]][classifiers_attr_str[2]] \
        == None:
        
        print 'Error, anger and irony model weights filename are required.\n' \
            % USAGE_STRING
        sys.exit(2)

    elif classifiers[classifiers_name_str[0]][classifiers_attr_str[3]] == None \
        or classifiers[classifiers_name_str[1]][classifiers_attr_str[3]] \
        == None or predic_distribution == None:
        
        print 'Error, anger, irony predict distribution paths are required.\n' \
            % USAGE_STRING
        sys.exit(2)
    else:
        for classifier_name, classifier_dict in classifiers.iteritems():
            classifier_dict[classifiers_attr_str[3]] = \
                ujson.load(open(classifier_dict[classifiers_attr_str[3]], 'r'))

            if max_phrase_length is None:
                max_phrase_length = classifier_dict[classifiers_attr_str[3]] \
                    ['max_phrase_length']
            else:
                if max_phrase_length != classifier_dict[ \
                    classifiers_attr_str[3] ]['max_phrase_length']:

                    print 'Error, both classifiers must be pre-trained with '
                    + 'same max_phrase_length.'
                    sys.exit(2)

    df = pd.read_csv(dataset_path, header=0, \
        dtype={COMPULSORY_COLUMNS[0]: np.int64})

    X_predict = prepare_samples(word2vec_dataset_path, max_phrase_length)

    predicted_distribution = {'class': {}, 'category': {}}

    for classifier_name, classifier_dict in classifiers.iteritems():

        # Load model weights path into the dictionary.
        classifier_dict[classifiers_attr_str[2]] = os.path.join( \
            classifier_dict[classifiers_attr_str[0]], 'model_weights/' \
            + classifier_dict[classifiers_attr_str[2]])

        model = load_model(classifier_dict[classifiers_attr_str[1]], \
            classifier_dict[classifiers_attr_str[2]])

        y_predict = model.predict([X_predict, X_predict, X_predict])
        y_predict = probas_to_classes(y_predict)

        df[classifier_name] = y_predict

        labels = classifier_dict[classifiers_attr_str[3]]['classes'].keys()

        for index in xrange(len(labels)):
            df.loc[df[classifier_name] == index, classifier_name] \
                = labels[index]

            predicted_distribution['class'][labels[index]] = len(df[ 
                (df[COMPULSORY_COLUMNS[1]] == labels[index]) \
                & (df[classifier_name] == df[COMPULSORY_COLUMNS[1]]) ].index)

        predicted_distribution['category'][classifier_name] = \
            len(df[df[classifier_name] == df[COMPULSORY_COLUMNS[1]]].index)


    for label, valid_instance_num in predic_distribution['classes'].iteritems():

        num_predicted_label = float(predicted_distribution['class'][label])

        print label + ' accuracy: ' + str(num_predicted_label \
            / valid_instance_num)

    print ''

    for label, num_predicted_label in predicted_distribution['category'] \
        .iteritems():

        print label + ' classifier accuracy: ' + str(num_predicted_label \
            / ( float(predic_distribution['classes'][label] \
            + predic_distribution['classes']['no_' + label]) ))

    print ''

    print 'Dataset length: ' + str(len(df.index))
    
    df[result_col] = None

    for label, condition in matrix_classes.iteritems():
        df.loc[((df[classifiers_name_str[0]] == condition[0]) \
            & (df[classifiers_name_str[1]] == condition[1])) \
            | ((df[classifiers_name_str[0]] == condition[1]) \
            & (df[classifiers_name_str[1]] == condition[0])), result_col] \
            = label

    print df

    sys.exit(0)


    if target == TARGETS[0] or target == TARGETS[2]:
        if train_path == None or validation_path == None:
            print 'Error: Train and Validation dataset paths are required. %s' \
                % USAGE_STRING
            sys.exit(2)
        else:
            dataset_name = train_path.rsplit('/', 1)[1].split('_train.json')[0]
            dataset_result_output_path = os.path.join(SCRIPT_DIR, \
                dataset_name)
            check_valid_dir(dataset_result_output_path)
            
            model_output_path = os.path.join(dataset_result_output_path, \
                'model')
            check_valid_dir(model_output_path)
            model_weights_output_path = os.path.join( \
                dataset_result_output_path, 'model_weights' )
            check_valid_dir(model_weights_output_path)

    if target == TARGETS[1] or target == TARGETS[2]:
        if test_path == None:
            print 'Error: Test dataset path is required. %s' % USAGE_STRING
            sys.exit(2)
        elif dataset_name is None:
            dataset_name = test_path.rsplit('/', 1)[1].split('_test.json')[0]
            dataset_result_output_path = os.path.join(SCRIPT_DIR, \
                dataset_name)
            check_valid_dir(dataset_result_output_path)

        if target == TARGETS[1] \
            and (model_path == None or weights_path == None):
            
            print 'Error: model and weights paths must be provided to ' \
             + 'execute the test. %s' % USAGE_STRING
            sys.exit(2)

    results_output_path = os.path.join(dataset_result_output_path, 'results')
    check_valid_dir(results_output_path)

    if distribution_path is None:
        if train_path is not None:
            distribution_path = train_path.rsplit('_train.json', 1)[0] \
                + '_distribution.json'
        else:
            distribution_path = test_path.rsplit('_test.json', 1)[0] \
                + '_distribution.json'

    class_distribution = ujson.load(open(distribution_path, 'r'))
    labels = [key for key in class_distribution['classes'].keys()]
    num_categories = len(class_distribution['classes'])
    max_phrase_length = class_distribution['max_phrase_length']
    model_size = class_distribution['model_feature_length']

    print "Model size: " + str(model_size)
    print "Number of filters: " + str(NUM_FILTERS)
    print "Batch size: " + str(BATCH_SIZE)
    print "Number of epochs: " + str(NB_EPOCH)
    print "Evaluation period: " + str(EVAL_PERIOD)
    print 'Max. phrase length: ' + str(max_phrase_length)
    print 'Building model...'

    if target == TARGETS[1]:
        merged = load_model(model_path, weights_path)
    else:
        merged = generate_model(model_size, max_phrase_length, num_categories)
    
    merged.summary()
    
    # Execute training if target is train or all.
    if target == TARGETS[0] or target == TARGETS[2]:
        train(train_path, validation_path, merged, labels, max_phrase_length, \
            model_output_path, model_weights_output_path)

    # Execute test if target is test or all.
    if target == TARGETS[1] or target == TARGETS[2]:
        test(test_path, merged, labels, max_phrase_length, results_output_path)

if __name__ == '__main__':
    start_time = datetime.now()
    main(sys.argv[1:])
    print "Elapsed time: " + str(datetime.now() - start_time)
