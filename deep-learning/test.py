# /usr/bin/env
# -*- coding: utf-8 -*-

from datetime import datetime
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Merge
from keras.models import Sequential
#import tensorflow
#from tensorflow.python.ops import control_flow_ops
#tensorflow.python.control_flow_ops = control_flow_ops
import time, os, gc, sys
import getopt
import ujson, json, base64
import numpy as np

NUM_FILTERS = 200
BATCH_SIZE = 50
NB_EPOCH = 1000
EVAL_PERIOD = 12

CWD = os.getcwd()

def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

def prepare_samples(piece_path, labels, max_phrase_length):
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
        X = X.reshape(X.shape[0], 1, max_phrase_length, 300)
        #input_shape = (1, img_rows, img_cols)
    else:
        print len(X[0])
        X = X.reshape(X.shape[0], max_phrase_length, 300, 1)
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

def load_model(model):
    model.load_weights('/home/jesus/anger-detection/deep-learning/experiment_withoutcontext_6.h5')
    return model


def save_model(model):
    json_string = model.to_json()
    model_name = 'experiment_withoutcontext_6'
    open(model_name + '.json', 'w').write(json_string)
    model.save_weights(model_name + '.h5', overwrite=True)

def check_valid_path(path, desc):
    if not os.path.isabs(path):
        # Make relative path absolute.
        path = os.path.join(CWD, path)

    if not os.path.isfile(path):
        print 'Error: Invalid %s file.' % desc
        sys.exit(2)

    return path

def init_training(argv):

    train_path = None
    test_path = None
    distribution_path = None

    try:
        opts, args = getopt.getopt(argv,'',['train=', 'test=', 'distribution='])
    except getopt.GetoptError:
        print 'ERROR: Unknown parameter. Usage: cnn.py ' \
            + '--test=path_to_test_word2vec ' \
            + '[--distribution=path_to_distribution_file]'
        sys.exit(2)

    for o, a in opts:
        if o == '--test':
            test_path = check_valid_path(a, 'test dataset')
        elif o == '--distribution':
            distribution_path = check_valid_path(a, 'distribution dataset')

    if distribution_path is None:
        distribution_path = train_path.rsplit('_train.json', 1)[0] \
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

    branch_ngram_3 = Sequential()

    branch_ngram_3.add(Convolution2D(NUM_FILTERS, 3, model_size, \
        input_shape=(max_phrase_length, model_size, 1), border_mode='valid', \
        activation='relu'))
    branch_ngram_3.add(MaxPooling2D(pool_size=(max_phrase_length - 3 , 1)))
    #branch_ngram_3.add(Flatten())

    branch_ngram_4 = Sequential()
    branch_ngram_4.add(Convolution2D(NUM_FILTERS, 4, model_size, \
        input_shape=(max_phrase_length, model_size, 1), border_mode='valid',  
        activation='relu'))
    branch_ngram_4.add(MaxPooling2D(pool_size=(max_phrase_length - 4 + 1, 1)))
    #branch_ngram_4.add(Flatten())

    branch_ngram_5 = Sequential()
    branch_ngram_5.add(Convolution2D(NUM_FILTERS, 5, model_size, \
        input_shape=(max_phrase_length, model_size, 1), border_mode='valid', \
        activation='relu'))
    branch_ngram_5.add(MaxPooling2D(pool_size=(max_phrase_length - 5 + 1, 1)))
    #branch_ngram_5.add(Flatten())

    merged = Sequential()
    # merged = Merge([branch_ngram_2, branch_ngram_3, branch_ngram_4], \
    # merge_mode='concat')

    merged.add(Merge([branch_ngram_3, branch_ngram_4, branch_ngram_5], \
        mode='concat', concat_axis=2))
    merged.add(Flatten())
    merged.add(Dropout(0.5))

    merged.add(Dense(num_categories))
    merged.add(Activation('softmax'))
    merged.compile(loss='categorical_crossentropy', optimizer='adam', \
        metrics=['accuracy', 'mse', 'mae'])
    merged.summary()
    
    merged = load_model(merged)

    evaluation = {}
    X_test, y_test = prepare_samples(test_path, labels, max_phrase_length)
    
    print 'Testing...'
    sys.stdout.flush()
    
    scores = merged.evaluate([X_test, X_test, X_test], y_test, verbose=0, batch_size=BATCH_SIZE)
    print("%s: %.2f%%" % (merged.metrics_names[1], scores[1]*100))

    print scores
    
    print 'LLAP'

if __name__ == '__main__':
    start_time = datetime.now()
    init_training(sys.argv[1:])
    print "Elapsed time: " + str(datetime.now() - start_time)
