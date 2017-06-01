# /usr/bin/env
# -*- coding: utf-8 -*-

from datetime import datetime
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Merge
from keras.layers.normalization import BatchNormalization
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

NUM_FILTERS = 200
BATCH_SIZE = 50
NB_EPOCH = 1000
EVAL_PERIOD = 12
PATIENCE = 20
STOP_CONDITION = 'val_loss'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()

# Hack to import from parent folder.
sys.path.append(SCRIPT_DIR + '/..')
from drawing.drawing_utils import EpochDrawer, ConfusionMatrixDrawer

TARGETS = ['train', 'test', 'all']

USAGE_STRING = 'Usage: cnn.py [-h] [--help] --target=[train, test, all] ' \
    + '--train=path_to_train_word2vec ' \
    + '--validation=path_to_validation_word2vect ' \
    + '[--test=path_to_test_word2vec]' \
    + '[--distribution=path_to_distribution_file] ' \
    + '[--load_model=path_to_model] [--load_weights=path_to_weights]'

def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

def prepare_samples(piece_path, labels, max_phrase_length, model_size):
    X = []
    y = []
    with open(piece_path, 'r') as piece:
        program = json.load(piece, object_hook=json_numpy_obj_hook)
       
        encoder = one_hot_encoder(labels)
        for phrase in program:
            X.append(np.array(phrase['word2vec']))
            y.append(encoder[phrase['label']])
    X = np.array(X)

    if K.image_dim_ordering() == 'th':
        X = X.reshape(X.shape[0], 1, max_phrase_length, model_size)
        #input_shape = (1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], max_phrase_length, model_size, 1)
        #input_shape = (img_rows, img_cols, 1)
    return X, y

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

def remove_layers(model, number):
    model.summary()
    print 'Number of layer before removing the last one' \
        + str(len(model.layers)) + ' and its output shape is ' \
        + str(model.layers[-1].output_shape)
    for i in range(number):
        model.layers.pop()

    print 'Number of layer after removing the last one' \
        + str(len(model.layers)) + ' and its output shape is ' \
        + str(model.layers[-1].output_shape)
    
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    
    print 'Number of layer after adding a fake dense' \
        + str(len(model.layers)) +' and its output shape is ' \
        + str(model.layers[-1].output_shape)
    model.compile(loss={'main': 'categorical_crossentropy'}, optimizer='adam', \
        metrics=['accuracy', 'mse', 'mae'])
    model.summary()
    return model

def generate_model(model_size, max_phrase_length, num_categories):
    branch_ngram_3 = Sequential()

    branch_ngram_3.add(Convolution2D(NUM_FILTERS, 3, model_size, \
        input_shape=(1, max_phrase_length, model_size), \
        border_mode='valid', activation='relu'))
    branch_ngram_3.add(MaxPooling2D(pool_size=(max_phrase_length - 3 , 1)))
    #branch_ngram_3.add(Flatten())

    branch_ngram_4 = Sequential()
    branch_ngram_4.add(Convolution2D(NUM_FILTERS, 4, model_size, \
        input_shape=(1, max_phrase_length, model_size), border_mode='valid',
        activation='relu'))
    branch_ngram_4.add(MaxPooling2D( \
        pool_size=(max_phrase_length - 4 + 1, 1)) )
    #branch_ngram_4.add(Flatten())

    branch_ngram_5 = Sequential()
    branch_ngram_5.add(Convolution2D(NUM_FILTERS, 5, model_size, \
        input_shape=(1, max_phrase_length, model_size), \
        border_mode='valid', activation='relu'))
    branch_ngram_5.add(MaxPooling2D( \
        pool_size=(max_phrase_length - 5 + 1, 1)) )
    #branch_ngram_5.add(Flatten())

    merged = Sequential()
    # merged = Merge([branch_ngram_2, branch_ngram_3, branch_ngram_4], \
    # merge_mode='concat')

    merged.add(Merge([branch_ngram_3, branch_ngram_4, branch_ngram_5], \
        mode='concat', concat_axis=2))
    merged.add(Flatten())
    merged.add(Dropout(0.5))

    # Add one or two Densa ReLu.
    merged.add(Dense(512, activation='relu'))
    # merged.add(Dense(512, activation='relu'))

    merged.add(Dense(num_categories))

    # Add batch normalization layer
    merged.add(BatchNormalization())

    merged.add(Activation('softmax'))
    merged.compile(loss='binary_crossentropy', optimizer='adam', \
        metrics=['accuracy', 'mse', 'mae'])

    return merged

def save_model(model, model_output_path, model_weights_output_path, model_size):
    json_string = model.to_json()
    open(model_output_path + '/model.json', 'w') \
        .write(json_string)
    model.save_weights(model_weights_output_path + '/model_weights-' + \
        str(model_size) +'.h5', overwrite=True)

def load_model(model_path, weights_path):
    model_file = open(model_path, 'r')
    model_str = model_file.read()
    model_file.close()

    model = model_from_json(model_str)
    model.load_weights(weights_path)
    model.compile(loss='binary_crossentropy', optimizer='adam', \
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

def train(train_path, validation_path, model, labels, max_phrase_length, \
    model_output_path, model_weights_output_path, model_size):
    
    X_train, y_train = prepare_samples(train_path, labels, max_phrase_length, \
        model_size)
    X_validation, y_validation = prepare_samples(validation_path, labels, \
        max_phrase_length, model_size)

    early_stopping = EarlyStopping(monitor=STOP_CONDITION, \
        min_delta=0, patience=PATIENCE, verbose=0, mode='auto')
    #TODO:Check how save_best_only works
    checkpoint = ModelCheckpoint(model_weights_output_path + '/{epoch:02d}-{' \
        + STOP_CONDITION + ':.2f}-' + str(model_size) + '.hdf5', \
        monitor=STOP_CONDITION, verbose=0, save_best_only=True, \
        save_weights_only=False, mode='auto')

    print 'Training...'
    sys.stdout.flush()

    model.fit([X_train, X_train, X_train], y_train, batch_size=BATCH_SIZE, \
        nb_epoch=NB_EPOCH, \
        validation_data=([X_validation, X_validation, X_validation], \
            y_validation), callbacks=[checkpoint, early_stopping])

    print 'Saving model...'
    sys.stdout.flush()

    save_model(model, model_output_path, model_weights_output_path, model_size)
    print 'LLAP'

def test(test_path, model, labels, max_phrase_length, output_path, model_size):
    # Using keras evaluation method,
    # just check if I am doing it right with SciKit.
    X_test, y_test = prepare_samples(test_path, labels, max_phrase_length, \
        model_size)

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

    with codecs.open(output_path + '/' + timestamp + '(' + str(model_size) + \
        ' features)', 'w', encoding='utf-8') as file:

        file.write(ujson.dumps(results, indent=4))

def main(argv):
    train_path = None
    validation_path = None
    test_path = None
    distribution_path = None
    model_path = None
    weights_path = None

    dataset_result_output_path = None
    model_output_path = None
    model_weights_output_path = None
    results_output_path = None

    target = TARGETS[2]
    merged = None
    dataset_name = None

    try:
        opts, args = getopt.getopt(argv,'h',['train=', 'validation=','test=', \
            'load_model=', 'load_weights=','distribution=', 'target=', 'help'])
    except getopt.GetoptError:
        print 'Error: Unknown parameter. %s' % USAGE_STRING
        sys.exit(2)

    for o, a in opts:
        if o == '-h' or o == '--help':
            print USAGE_STRING
            sys.exit(0)
        elif o == '--train':
            train_path = check_valid_path(a, 'train dataset')
        elif o == '--validation':
            validation_path = check_valid_path(a, 'validation dataset')
        elif o == '--test':
            test_path = check_valid_path(a, 'test dataset')
        elif o == '--distribution':
            distribution_path = check_valid_path(a, 'distribution dataset')
        elif o == '--load_model':
            model_path = check_valid_path(a, 'model')
        elif o == '--load_weights':
            weights_path = check_valid_path(a, 'weights')
        elif o == '--target':
            valid_target = False
            for value in TARGETS:
                if a.lower() == value:
                    target = value
                    valid_target = True
                    break

            if not valid_target:
                print 'Error: Unknown target, specified a valid one from: %s' \
                    % str(TARGETS).strip('()')
                sys.exit(2)

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
            model_output_path, model_weights_output_path, model_size)

    # Execute test if target is test or all.
    if target == TARGETS[1] or target == TARGETS[2]:
        test(test_path, merged, labels, max_phrase_length, \
            results_output_path, model_size)

if __name__ == '__main__':
    start_time = datetime.now()
    main(sys.argv[1:])
    print "Elapsed time: " + str(datetime.now() - start_time)


################################################################################

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
from collections import OrderedDict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()

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
            X.append(np.array(phrase['word2vec']))
    X = np.array(X)

    if K.image_dim_ordering() == 'th':
        X = X.reshape(X.shape[0], 1, max_phrase_length, model_length)
        #input_shape = (1, img_rows, img_cols)
    else:
        X = X.reshape(X.shape[0], max_phrase_length, model_length, 1)
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
        y_predict = model.predict([X_predict, X_predict, X_predict])
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
