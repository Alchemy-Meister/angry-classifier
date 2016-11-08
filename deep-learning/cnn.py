from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Merge
from keras.models import Sequential
from utils import prepare_samples
#import tensorflow
#from tensorflow.python.ops import control_flow_ops
#tensorflow.python.control_flow_ops = control_flow_ops
import time, os, gc, sys
import ujson

MODEL_SIZE = 200
NUM_FILTERS = 200
BATCH_SIZE = 50
NB_EPOCH = 500
DATASET_PATH = './../datasets/crowdflower/json/crowdflower'
EVAL_PERIOD = 12
MAX_PHRASE_LENGTH = 31

def save_model(model):
    json_string = model.to_json()
    model_name = 'experiment_withoutcontext_6'
    open(model_name + '.json', 'w').write(json_string)
    model.save_weights(model_name + '.h5', overwrite=True)


def init_training(num_categories, dataset_path):
    print "Model size:" + str(MODEL_SIZE)
    print "Number of filters: " + str(NUM_FILTERS)
    print "Batch size: " + str(BATCH_SIZE)
    print "Number of epochs: " + str(NB_EPOCH)
    print "Class to classify: " + CLASS_TO_CLASSIFY
    print "Evaluation period: " + str(EVAL_PERIOD)
    print 'Building model...'
    
    branch_ngram_3 = Sequential()
    branch_ngram_3.add(Convolution2D(NUM_FILTERS, 3, MODEL_SIZE, \
        input_shape=(1, MAX_PHRASE_LENGTH, MODEL_SIZE), border_mode='valid', \
        activation='relu'))
    branch_ngram_3.add(MaxPooling2D(pool_size=(MAX_PHRASE_LENGTH - 3 + 1 , 1)))
    #branch_ngram_3.add(Flatten())

    branch_ngram_4 = Sequential()
    branch_ngram_4.add(Convolution2D(NUM_FILTERS, 4, MODEL_SIZE, \
        input_shape=(1, MAX_PHRASE_LENGTH, MODEL_SIZE), border_mode='valid',  
        activation='relu'))
    branch_ngram_4.add(MaxPooling2D(pool_size=(MAX_PHRASE_LENGTH - 4 + 1, 1)))
    #branch_ngram_4.add(Flatten())

    branch_ngram_5 = Sequential()
    branch_ngram_5.add(Convolution2D(NUM_FILTERS, 5, MODEL_SIZE, \
        input_shape=(1, MAX_PHRASE_LENGTH, MODEL_SIZE), border_mode='valid', \
        activation='relu'))
    branch_ngram_5.add(MaxPooling2D(pool_size=(MAX_PHRASE_LENGTH - 5 + 1, 1)))
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
    
    evaluation = {}
    X_train, y_train = prepare_samples(dataset_path + '_train.json')
    X_test, y_test = prepare_samples(dataset_path + '_test.json')
    
    print 'Training...'
    sys.stdout.flush()
    
    merged.fit([X_train, X_train, X_train], y_train, batch_size=50, \
        nb_epoch=1000, validation_data=([X_test, X_test, X_test], y_test))
    
    print 'Saving model...'
    sys.stdout.flush()
    
    save_model(merged)
    print 'LLAP'

if __name__ == '__main__':
    class_distribution = open(DATASET_PATH + '_distribution.json', 'r')
    num_categories = len(ujson.loads(class_distribution))
    init_training(num_categories, DATASET_PATH)
