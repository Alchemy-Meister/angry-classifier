# /usr/bin/env
# -*- coding: utf-8 -*-

from datetime import datetime
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import Convolution1D, Convolution2D, GlobalMaxPooling1D, \
    MaxPooling2D, merge, Merge, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
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

FILTSZ = [3, 4, 5]
NUM_FILTERS = 200
BATCH_SIZE = 50
NB_EPOCH = 1000
EVAL_PERIOD = 12
PATIENCE = 20
STOP_CONDITION = 'val_loss'
DROPOUT = 0.5
BATCH_NORMALIZATION = False
BINARY = True
BATCH_NORMALIZATION_RELU_SOFT = False
DENSES = ['relu']

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()

# Hack to import from parent folder.
sys.path.append(SCRIPT_DIR + '/..')
from drawing.drawing_utils import EpochDrawer, ConfusionMatrixDrawer

TARGETS = ['train', 'test', 'all']

USAGE_STRING = 'Usage: cnn.py [-h] [--help] --target=[train, test, all] ' \
    + '--embedding_weights=path_to_embedding_weights' \
    + '--train=path_to_train_word2vec ' \
    + '--validation=path_to_validation_word2vect ' \
    + '--test=path_to_test_word2vec' \
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
            X.append(np.array(phrase['words']))
            y.append(encoder[phrase['label']])
    X = np.array(X)
    X = pad_sequences(X, maxlen=max_phrase_length, padding='post')
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

def generate_embedding_matrix(weights, max_phrase_length, trainable):
    return Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], \
        weights=[weights],  input_length=max_phrase_length, trainable=trainable)

def generate_model(model_size, max_phrase_length, num_categories, \
    embedding_weights, batch_normalization, dropout, trainable):

    x_1 = Input(shape=(max_phrase_length,), dtype='int32', name='phrases')

    embd = generate_embedding_matrix(embedding_weights, max_phrase_length, \
        trainable) (x_1)

    joined = generate_parallel_convolutionals(FILTSZ, embd, NUM_FILTERS, \
        max_phrase_length, model_size)

    if batch_normalization:
        batch_norm = BatchNormalization()(joined)
        drop = Dropout(dropout)(batch_norm)
    else:
        drop = Dropout(dropout)(joined)
        # joined.add(Dropout(dropout))
        # drop = joined
    dense = generate_second_part_after_cnns(drop, dropout, 'main', \
        DENSES, batch_normalization, BATCH_NORMALIZATION_RELU_SOFT, \
        BINARY, 'softmax')

    # Add one or two Densa ReLu.
    # merged.add(Dense(512, activation='relu'))
    # merged.add(Dense(512, activation='relu'))

    input_weights = [x_1]
    output_weights = [dense]

    # Add batch normalization layer
    # merged.add(BatchNormalization())

    print "Len the inputs " + str(len(input_weights))
    print "Len de outputs es " + str(len(output_weights))
    model = Model(input=input_weights, output=output_weights)
    model.compile(loss='binary_crossentropy', optimizer='adam', \
        metrics=['accuracy', 'mse', 'mae'])
    return model

def generate_old_parallel_convolutionals(filtsz, embed, num_filters, \
    max_phrase_length, model_size):
    
    convs = []
    joined = Sequential()
    for i, fsz in enumerate(filtsz):
        branch_ngram = Sequential()
        branch_ngram.add(Convolution2D(num_filters, fsz, model_size, \
            input_shape=(1, max_phrase_length, model_size), \
            border_mode='valid', activation='relu'))
        branch_ngram.add(MaxPooling2D( \
            pool_size=(max_phrase_length - fsz + 1, 1)) )
        convs.append(branch_ngram)
    
    joined.add(Merge(convs, mode='concat', concat_axis=2))
    joined.add(Flatten())
    
    return joined


def generate_parallel_convolutionals(filtsz, embed, num_filters, \
    max_phrase_length, model_size):
    
    convs = []
    joined = ""
    if len (filtsz) > 1:
        for i, fsz in enumerate(filtsz):
            conv = Convolution1D(num_filters, fsz, activation='relu', \
                input_length=max_phrase_length)(embed)
            gmp = GlobalMaxPooling1D()(conv)
            convs.append(gmp)
        joined = merge(convs, mode='concat')
    else:
        conv = Convolution1D(num_filters, filtsz[0], activation='relu', \
            input_length=max_phrase_length)(embed)
        joined = GlobalMaxPooling1D()(conv)

    return joined

def generate_second_old_part_after_cnns(drop1, dropout, name, denses, \
    batch_normalization, batch_normalization_relu_soft, binary, last_function):
    
    if denses[0] == "linear":
        dense_relu = Dense(512)(drop1)
    else:
        drop1.add(Dense(512, activation=denses[0]))
    # for dense in denses[1:]:
    #     drop = Dropout(dropout)(dense_relu)
    #     dense_relu = Dense(512, activation=dense)(drop)
    # if batch_normalization_relu_soft:
    #     dense_relu = BatchNormalization()(dense_relu)
    # print "Is binary?" + str(binary)
    if binary:
        softmax_len = 2
    # print "Last activation is " + last_function
    drop1.add(Dense(softmax_len, activation=last_function, name=name))
    return drop1

def generate_second_part_after_cnns(drop1, dropout, name, denses, \
    batch_normalization, batch_normalization_relu_soft, binary, last_function):
    
    if denses[0] == "linear":
        dense_relu = Dense(512)(drop1)
    else:
        dense_relu = Dense(512, activation=denses[0])(drop1)
    # for dense in denses[1:]:
    #     drop = Dropout(dropout)(dense_relu)
    #     dense_relu = Dense(512, activation=dense)(drop)
    # if batch_normalization_relu_soft:
    #     dense_relu = BatchNormalization()(dense_relu)
    # print "Is binary?" + str(binary)
    if binary:
        softmax_len = 2
    # print "Last activation is " + last_function
    dense = Dense(softmax_len, activation=last_function, name=name)(drop1)
    return dense

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

    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, \
        validation_data=(X_validation, y_validation), callbacks=[checkpoint, \
        early_stopping])

    print 'Saving model...'
    sys.stdout.flush()

    save_model(model, model_output_path, model_weights_output_path, model_size)
    print 'LLAP'

def test(test_path, model, labels, max_phrase_length, output_path, model_size):
    # Using keras evaluation method,
    # just check if I am doing it right with SciKit.
    X_test, y_test = prepare_samples(test_path, labels, max_phrase_length, \
        model_size)

    y_predict = model.evaluate(X_test, y_test, verbose=0, \
        batch_size=BATCH_SIZE)

    print 'The metrics keras is evaluating are: ' + str(model.metrics_names) \
    + ' and its results: ' + str(y_predict)
    # Using Scikit in order to evaluate the model.
    y_predict = model.predict(X_test)
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
    embedding_weights = None

    dataset_result_output_path = None
    model_output_path = None
    model_weights_output_path = None
    results_output_path = None

    target = TARGETS[2]
    merged = None
    dataset_name = None

    trainable = False

    try:
        opts, args = getopt.getopt(argv,'h',['embedding_weights=', 'train=', \
            'validation=','test=', 'load_model=', 'load_weights=', \
            'distribution=', 'target=', 'help'])
    except getopt.GetoptError:
        print 'Error: Unknown parameter. %s' % USAGE_STRING
        sys.exit(2)

    for o, a in opts:
        if o == '-h' or o == '--help':
            print USAGE_STRING
            sys.exit(0)
        elif o == '--embedding_weights':
            embedding_weights = check_valid_path(a, 'embeddings weights')
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
            weights_path = check_valid_path(a, 'model weights')
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
        if train_path == None or validation_path == None or embedding_weights \
            == None:
            
            print 'Error: Embedding weights and Train and Validation dataset' \
                + 'paths are required. %s' % USAGE_STRING
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

            embedding_weights = np.load(embedding_weights)

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
    print 'Embeddding trainables:' + str(trainable)
    print 'Building model...'

    if target == TARGETS[1]:
        merged = load_model(model_path, weights_path)
    else:
        merged = generate_model(model_size, max_phrase_length, num_categories, \
            embedding_weights, BATCH_NORMALIZATION, DROPOUT, trainable)
    
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
