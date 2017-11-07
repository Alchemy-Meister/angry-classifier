# /usr/bin/env
# -*- coding: utf-8 -*-

from datetime import datetime
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Embedding, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Merge, merge, Input
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical, probas_to_classes
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    precision_score, recall_score
from sklearn.utils import class_weight
from keras.optimizers import Adam

import codecs
import time, os, gc, sys
import getopt
import ujson, json, base64
import numpy as np
import pandas as pd
from collections import OrderedDict
import distutils.dir_util
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CWD = os.getcwd()

# Hack to import from parent folder.
sys.path.append(SCRIPT_DIR + '/..')
from drawing.drawing_utils import EpochDrawer, ConfusionMatrixDrawer

TARGETS = ['train', 'test', 'all']

USAGE_STRING = 'Usage: repressed-anger.py [-h] [--help] ' \
    + '--train=path_to_train_word2vec [--target=[train, test, all]]' \
    + '--evaluation=path_to_evaluation_word2vec' \
    + '--test=path_to_test_word2vec' \
    + '--test_dataset=path_to_original_dataset' \
    + '--dataset_distribution=path_to_dataset_dist' \
    + '[--anger_dir=path_to_anger_dir] ' \
    + '[--anger_distribution=anger_distribution_path]' \
    + '[--irony_dir=path_to_irony_dir] ' \
    + '[--irony-distribution=irony_distribution_path]' \
    + '--name=experiment_name -a --all_experiments' \
    + '-n --no_pretraining -f --freeze_branches' \
    + '-s --summary_only -t --trim_denses -c --clean_result_dir' \
    + '-w --use_class_weight' \
    + '-x --experimental' \
    + '-o --custom_optimizer'

CSV_COLUMNS = ['tweet_id', 'author', 'content', 'manual_label', 'label_1', \
    'label_2']
CSV_LABELS = CSV_COLUMNS[-2:]

# Author column is not required.
COMPULSORY_COLUMNS = list(CSV_COLUMNS)
del COMPULSORY_COLUMNS[1]

BATCH_SIZE = 50
NB_EPOCH = 1000
PATIENCE = 20
STOP_CONDITION = 'val_loss'
WEIGHT_FREESE = True
AGGREGATED_DENSE = True
AGGREGATED_DROPOUT = 0.4
NUM_EXECUTIONS = 25
 
LEARNING_RATE = 0.00001
DECAY = 0.0005

def json_numpy_obj_hook(dct):
    """Decodes a previously encoded numpy ndarray with proper shape and dtype.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

def prepare_samples(piece_path, classes, max_phrase_length, df):
    X = []
    y = []
    dataset_labels = []

    with open(piece_path, 'r') as piece:
        program = json.load(piece, object_hook=json_numpy_obj_hook)

        encoder = one_hot_encoder(classes)

        for indx, phrase in enumerate(program):
            X.append(np.array(phrase['words']))    
            y.append(encoder[df.get_value(indx, 'manual_label')])
            dataset_labels.append(df.get_value(indx, 'manual_label'))

    X = np.array(X)
    X = pad_sequences(X, maxlen=max_phrase_length, padding='post')
    return X, y, dataset_labels

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

def save_model(model, model_output_path, model_weights_output_path):
    json_string = model.to_json()
    open(model_output_path + '/model.json', 'w') \
        .write(json_string)
    model.save_weights(model_weights_output_path + '/model_weights' \
        +'.h5', overwrite=True)

def load_model(model_path, weights_path, load_pretrain_weights):
    model_file = open(model_path, 'r')
    model_str = model_file.read()
    model_file.close()

    model = model_from_json(model_str)
    print 'Load weights: %s' % load_pretrain_weights
    if load_pretrain_weights:
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
        print path
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

def list_directories(dir_path):
    lst = os.listdir(dir_path)
    lst.sort()
    return lst

def remove_files_dir(dir_path):
    for the_file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, the_file)
        try:
            if os.path.isfile(file_path):
                print file_path
                #os.unlink(file_path)
        except Exception as e:
            print(e)

def best_model_weights(dir_path):
    dir_list = list_directories(dir_path)
    return dir_list[-2]

def remove_layers(model, number):
    # model.summary()
    print 'Number of layer before removing the last one ' \
        + str(len(model.layers)) + ' and its output shape is ' \
        + str(model.layers[-1].output_shape)
    for i in range(number):
        model.layers.pop()

    print 'Number of layer after removing the last one ' \
        + str(len(model.layers)) + ' and its output shape is ' \
        + str(model.layers[-1].output_shape)
    
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    
    print 'Number of layer after adding a fake dense ' \
        + str(len(model.layers)) +' and its output shape is ' \
        + str(model.layers[-1].output_shape)
    model.compile(loss={'main': 'categorical_crossentropy'}, optimizer='adam', \
        metrics=['accuracy', 'mse', 'mae'])
    # model.summary()
    return model

def remove_denses(model):
    merge_layer_index = 1
    for layer in reversed(model.layers):
        if type(layer) is Merge:
            break
        merge_layer_index += 1

    merge_layer_index -= 1
    return remove_layers(model, merge_layer_index)

def train(train_path, validation_path, model, classes, max_phrase_length, \
    model_output_path, model_weights_output_path, dfs, use_class_weight, \
    experiment_results_output_path):
    
    X_train, y_train, df_classes = prepare_samples(train_path, classes, \
        max_phrase_length, dfs[0])

    class_weight_values = None
    if use_class_weight:
       class_weight_values = class_weight.compute_class_weight('balanced', \
        classes, df_classes)

    X_validation, y_validation, useless_var = prepare_samples(validation_path, \
        classes, max_phrase_length, dfs[1])

    early_stopping = EarlyStopping(monitor=STOP_CONDITION, \
        min_delta=0, patience=PATIENCE, verbose=0, mode='auto')
    checkpoint = ModelCheckpoint(model_weights_output_path + '/{epoch:02d}-{' \
        + STOP_CONDITION + ':.2f}' + '.hdf5', \
        monitor=STOP_CONDITION, verbose=0, save_best_only=True, \
        save_weights_only=False, mode='auto')

    print 'Training...'
    sys.stdout.flush()

    train_history = model.fit([X_train, X_train], y_train, \
        batch_size=BATCH_SIZE, class_weight=class_weight_values, \
        nb_epoch=NB_EPOCH, validation_data=([X_validation, X_validation], \
            y_validation), \
        callbacks=[checkpoint, early_stopping])

    check_valid_dir(experiment_results_output_path + '/epoch_plot/')
    EpochDrawer(train_history, save_filename=experiment_results_output_path \
        + '/epoch_plot/')

    print 'Saving model...'
    sys.stdout.flush()

    save_model(model, model_output_path, model_weights_output_path)
    print 'LLAP'

def test(test_path, model, classes, max_phrase_length, output_path, dfs):
    # Using keras evaluation method,
    # just check if I am doing it right with SciKit.
    X_test, y_test, useless_var = prepare_samples(test_path, classes, \
        max_phrase_length, dfs[2])

    y_predict = model.evaluate([X_test, X_test], y_test, verbose=0, \
        batch_size=BATCH_SIZE)

    print 'The metrics keras is evaluating are: ' + str(model.metrics_names) \
    + ' and its results: ' + str(y_predict)
    # Using Scikit in order to evaluate the model.

    y_predict = model.predict([X_test, X_test])

    y_predict = probas_to_classes(y_predict)

    y_raw_test = []
    for y1 in y_test:
        y_raw_test.append(find_1(y1))

    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)

    #Y_true, y_predict
    conf_matrix = confusion_matrix(y_raw_test, y_predict)
    ConfusionMatrixDrawer(conf_matrix, classes=classes, str_id=timestamp, \
        title='Confusion matrix, without normalization', folder=output_path)
    ConfusionMatrixDrawer(conf_matrix, classes=classes, normalize=True, \
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

    with codecs.open(output_path + '/' + timestamp, \
        'w', encoding='utf-8') as file:

        file.write(ujson.dumps(results, indent=4))

    return results

def main(argv):
    train_path = None
    validation_path = None
    test_path = None
    distribution_path = None
    dataset_name = None

    anger_dir = None
    irony_dir = None

    dataset_result_output_path = None
    results_output_path = None

    datasets = []
    dfs = []
    labels = []

    target = TARGETS[2]
    
    execute_all_experiments = False
    load_pretrain_weights = True
    freeze_branches = False
    trim_denses = False
    use_class_weight = False
    summary_only = False
    clean_result_dir = False
    use_custom_adam = False

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

    experiment_name = None
    subexperiment_name = 'direct'

    try:
        opts, args = getopt.getopt(argv,'acfhnostwx',['dataset=', 'train=', \
            'validation=', 'test=', 'target=','anger_dir=', \
            'anger_distribution=','irony_dir=', 'irony_distribution=', \
            'all_experiments', 'name=', 'help', 'distribution=', \
            'no_pretraining', 'freeze_branches', 'summary_only', \
            'trim_denses', 'clean_result_dir', 'use_class_weight', \
            'experimental', 'custom_optimizer'])
    except getopt.GetoptError:
        print 'Error: Unknown parameter. %s' % USAGE_STRING
        sys.exit(2)

    for o, a in opts:
        if o == '-h' or o == '--help':
            print USAGE_STRING
            sys.exit(0)
        if o == '--name':
            experiment_name = a
        elif o == '--dataset':
            for dataset_type in ['train', 'validation', 'test']:
                datasets.append(check_valid_path(a + '_' + dataset_type \
                    + '.csv', 'csvs'))

            for dataset in datasets:
                # Load original dataset.
                df = pd.read_csv(dataset, header=0, \
                    dtype={COMPULSORY_COLUMNS[0]: np.int64})

                # Maintain only manually labeled tweets.
                df = df[~df[COMPULSORY_COLUMNS[3]].isnull()]
                dfs.append(df)

                labels.extend(df.manual_label.unique().tolist())

            labels = list(set(labels))
        elif o == '--train':
            train_path = check_valid_path(a, 'train dataset')
        elif o == '--validation':
            validation_path = check_valid_path(a, 'validation dataset')
        elif o == '--test':
            test_path = check_valid_path(a, 'test dataset')
        elif o == '--distribution':
            distribution_path = check_valid_path(a, 'distribution dataset')
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
        elif o == '--anger_dir':
            anger_dir = check_valid_dir(a)
        elif o == '--anger_distribution':
            classifiers[classifiers_name_str[0]][classifiers_attr_str[3]] \
                = check_valid_path(a, 'anger distribution')
        elif o == '--irony_dir':
            irony_dir = check_valid_dir(a)
        elif o == '--irony_distribution':
            classifiers[classifiers_name_str[1]][classifiers_attr_str[3]] \
                = check_valid_path(a, 'irony distribution')
        elif o == '--all_experiments' or o == '-a':
            execute_all_experiments = True
        elif o == '--no_pretraining' or o == '-n':
            load_pretrain_weights = False
            subexperiment_name = 'no_pretraining'
        elif o == '--freeze_branches' or o == '-f':
            subexperiment_name = subexperiment_name + '_freezed'
            freeze_branches = True
        elif o == '--summary_only' or o == '-s':
            summary_only = True
        elif o == '--trim_denses' or 0 == '-t':
            trim_denses = True
            subexperiment_name = subexperiment_name + '_denses_trimmed'
        elif o == '--clean_result_dir' or o == '-c':
            clean_result_dir = True
        elif o == '--use_class_weight' or o == '-w':
            use_class_weight = True
            subexperiment_name = subexperiment_name + '_class_weights'
        elif o == '--experimental' or o == '-x':
            subexperiment_name = subexperiment_name + '_experimental'
        elif o == '--custom_optimizer' or o == '-o':
            subexperiment_name = subexperiment_name + '_adam_lr_' \
            + str(LEARNING_RATE) + '_dec_' + str(DECAY)
            use_custom_adam = True



    print 'experiment name: %s' % subexperiment_name

    # Input error detection.
    if anger_dir == None or irony_dir == None:
        print 'Error: anger and irony directories are required.\n %s' \
            % USAGE_STRING
        sys.exit(2)

    elif classifiers[classifiers_name_str[0]][classifiers_attr_str[3]] == None \
        or classifiers[classifiers_name_str[1]][classifiers_attr_str[3]] \
        == None or distribution_path == None:
        
        print 'Error: anger, irony predict distribution paths are required.' \
            + '\n%s' % USAGE_STRING
        sys.exit(2)
    else:
        distribution = ujson.load(open(distribution_path, 'r'))
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

        if max_phrase_length != distribution['max_phrase_length']:
            print 'Error: prediction and pre-trained word embeddings must ' \
                + 'have same max_phrase_length.'
            sys.exit(2)
        elif model_length != distribution['model_feature_length']:
            print 'Error: prediction and pre-trained word embeddings must ' \
                + 'have same model_feature_length.'
            sys.exit(2)

    if target == TARGETS[0] or target == TARGETS[2]:
        if train_path == None or validation_path == None \
            == None:

            print 'Error: Train and Validation dataset' \
                + 'paths are required. %s' % USAGE_STRING
            sys.exit(2)
        else:
            dataset_name = train_path.rsplit('/', 1)[1].split('_train.json')[0]
            dataset_result_output_path = os.path.join(SCRIPT_DIR, \
                dataset_name)
            check_valid_dir(dataset_result_output_path)

    if execute_all_experiments:
        experiments = list_directories( os.path.join(anger_dir, \
            'model/backup_deusto/embeddings/') )
    else:
        experiments = [experiment_name]

    for experiment_name in experiments:

        model_output_path = os.path.join(dataset_result_output_path, \
            'model/embeddings/second_phase/' + experiment_name + '/' \
            + subexperiment_name)
        check_valid_dir(model_output_path)
        
        model_weights_output_path = os.path.join( \
            dataset_result_output_path, 'model_weights' \
            + '/embeddings/second_phase/' + experiment_name + '/' \
            + subexperiment_name)
        check_valid_dir(model_weights_output_path)
        
        results_output_path = os.path.join(dataset_result_output_path, \
            'results' + '/embeddings/second_phase/' + experiment_name + '/' \
            + subexperiment_name)
        check_valid_dir(results_output_path)

        models = []

        classifiers[classifiers_name_str[0]][classifiers_attr_str[0]] \
            = anger_dir
        classifiers[classifiers_name_str[0]][classifiers_attr_str[1]] \
            = os.path.join(anger_dir, 'model/backup_deusto/embeddings/' \
            + experiment_name + '/model.json')
        classifiers[classifiers_name_str[1]][classifiers_attr_str[0]] \
            = irony_dir
        classifiers[classifiers_name_str[1]][classifiers_attr_str[1]] \
            = os.path.join(irony_dir, 'model/backup_deusto/embeddings/' \
            + experiment_name + '/model.json')

        for ndx, (classifier_name, classifier_dict) in enumerate( \
            classifiers.iteritems(), 1):

            # Load model weights path into the dictionary.
            model_weights_dir = os.path.join( \
                classifier_dict[classifiers_attr_str[0]], 'model_weights/' \
                + 'backup_deusto/embeddings/' + experiment_name + '/')

            model_weights_file = best_model_weights(model_weights_dir)

            classifier_dict[classifiers_attr_str[2]] = os.path.join( \
                classifier_dict[classifiers_attr_str[0]], 'model_weights/' \
                + 'backup_deusto/embeddings/' + experiment_name + '/' \
                + model_weights_file)

            model = load_model(classifier_dict[classifiers_attr_str[1]], \
                classifier_dict[classifiers_attr_str[2]], load_pretrain_weights)
            
            if trim_denses:
                # Removes all layers after conv merge
                model = remove_denses(model)
            else:
                # Removes only softmax layer of the model
                model = remove_layers(model, 1)

            print 'Freezing branch: %s' % freeze_branches
            # Freeze branches
            if freeze_branches:
                for layer in model.layers:
                    layer.trainable = False

            for layer in model.layers:
                model.get_layer(name=layer.name).name = layer.name + '_' \
                + str(ndx)

            models.append(model)

        agregated = merge([x.layers[-1].output for x in models], mode='concat')

        if AGGREGATED_DENSE:
            agregated = Dense(512, activation='relu')(agregated)
            agregated = Dropout(AGGREGATED_DROPOUT)(agregated)

        agregated = Dense(4, activation='softmax', name='agregated_softmax') \
            (agregated)

        aggregated_model = Model( input=[x.input for x in models], \
            output=[agregated] )

        if use_custom_adam:
            custom_adam = Adam(lr=LEARNING_RATE, decay=DECAY)
            aggregated_model.compile( loss='categorical_crossentropy', \
                optimizer=custom_adam, metrics=['accuracy', 'mse', 'mae'] )
        else:
            aggregated_model.compile( loss='categorical_crossentropy', \
                optimizer='adam', metrics=['accuracy', 'mse', 'mae'] )

        aggregated_model.summary()

        if not summary_only:
            results = []
            best_result = None

            for experiment_number in xrange(NUM_EXECUTIONS):
                experiment_model_output = model_output_path + '/' \
                    + str(experiment_number)
                experiment_model_weights_output = model_weights_output_path \
                    + '/' + str(experiment_number)
                experiment_results_output_path = results_output_path + '/' \
                    + str(experiment_number)
                check_valid_dir(experiment_model_output)
                check_valid_dir(experiment_model_weights_output)
                check_valid_dir(experiment_results_output_path)

                if clean_result_dir:
                    remove_files_dir(experiment_model_output)
                    remove_files_dir(experiment_model_weights_output)
                    remove_files_dir(experiment_results_output_path)

                # Execute training if target is train or all.
                if target == TARGETS[0] or target == TARGETS[2]:
                    train(train_path, validation_path, aggregated_model, \
                        labels, max_phrase_length, experiment_model_output, \
                        experiment_model_weights_output, dfs, \
                        use_class_weight, experiment_results_output_path)

                # Execute test if target is test or all.
                if target == TARGETS[1] or target == TARGETS[2]:
                    results.append(test(test_path, aggregated_model, labels, \
                        max_phrase_length, experiment_results_output_path, dfs))

                if best_result == None:
                    best_result = results[0]
                    best_result['experiment_no'] = 0
                elif best_result['f1_macro'] \
                    < results[experiment_number]['f1_macro']:
                    
                    best_result = results[experiment_number]
                    best_result['experiment_no'] = experiment_number

            # Copy best result experiment to best/ folder.
            distutils.dir_util.copy_tree( model_output_path + '/' \
                    + str(best_result['experiment_no']), model_output_path \
                    + '/best/' )
            distutils.dir_util.copy_tree( model_weights_output_path + '/' \
                    + str(best_result['experiment_no']), \
                    model_weights_output_path + '/best/' )
            distutils.dir_util.copy_tree( results_output_path + '/' \
                    + str(best_result['experiment_no']), results_output_path \
                    + '/best/' )

            # Calculates average results
            avg_result = {}
            for result in results:
                for key, value in result.iteritems():
                    if key not in avg_result:
                        avg_result[key] = value
                    else:
                        avg_result[key] += value
            
            for key, value in avg_result.iteritems():
                avg_result[key] /= len(results)
            
            with codecs.open(results_output_path + '/' \
                + '/best/avg_result.json', 'w', encoding='utf-8') as file:

                file.write(ujson.dumps(avg_result, indent=4))

if __name__ == '__main__':
    start_time = datetime.now()
    main(sys.argv[1:])
    print "Elapsed time: " + str(datetime.now() - start_time)
