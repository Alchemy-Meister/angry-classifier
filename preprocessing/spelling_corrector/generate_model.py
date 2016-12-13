# /usr/bin/env
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import codecs
from collections import Counter
from datetime import datetime
import getopt
import math
import os
from tqdm import trange, tqdm
import re
import sys
import ujson
import uuid

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'language_model/')
BOOK_DIR = os.path.join(MODEL_DIR, 'project_gutenberg/A/')
OXFORD_DIR = os.path.join(MODEL_DIR, 'oxford/')
OPEN_SUBS_DIR = os.path.join(MODEL_DIR, 'open_subtitles/')
WIKTIONARY_DIR = os.path.join(MODEL_DIR, 'wiktionary/')
FILE_LIST_INFO_FILENAME = 'model_file_list_info.json'
OUTPUT_FILENAME = 'english (wf)'

def save_file_info_dict(dir_list, path, wordfreq_path):
    # Append file info to the list by reference.
    dir_list.append({'path': path, 'word_frequency': wordfreq_path})

def update_file_list_info(clean):
    # Updates info file list file from where the model is created.
    if os.path.exists(BOOK_DIR):
        
        files = []
        utf_files = []
        dir_desc = []
        dir_list = [OXFORD_DIR, OPEN_SUBS_DIR, WIKTIONARY_DIR, BOOK_DIR]
        clean_dir_index = 3

        # Generate a list of basename string of the directories.
        for path in dir_list:
            dirs = path.split('/')
            dir_desc.append(dirs[len(dirs) - 2])

        for index, directiory in enumerate(dir_list):
            file_list = os.listdir(directiory)
            file_list_length = len(file_list)

            for file_index in trange(file_list_length, \
                desc='Generating ' + dir_desc[index] 
                + ' directory info', total=file_list_length):
            
                #Gets absolute directory to the file.
                filename = os.path.join(directiory, file_list[file_index])
            
                # Excludes non files and irrelevant.
                if os.path.isfile(filename) and '(cover)' not in filename:
                    # Search non for word frequency files.
                    if '(wf)' not in filename:
                        # Add file to the info list.
                        if clean or not os.path.exists(filename + ' (wf)'):
                            save_file_info_dict(files, filename, filename \
                                + ' (wf)')

                            save_file_info_dict(utf_files, \
                                filename.decode('utf-8'), \
                                (filename + ' (wf)').decode('utf-8'))

                    elif index != clean_dir_index or not clean:
                        # Save word frequency file to the list.
                        save_file_info_dict(files, filename, filename)

                        save_file_info_dict(utf_files, \
                            filename.decode('utf-8'), filename.decode('utf-8'))
                    elif clean and index == clean_dir_index:
                        # Remove word frequency files from storage.
                        os.remove(filename)

    # Serializes info list into a file.
    with codecs.open(MODEL_DIR + FILE_LIST_INFO_FILENAME, 'w', \
        encoding='utf-8') as ofile:
        
        ofile.write(ujson.dumps(utf_files, indent=4))

    return files

# Regular expression used to split text from the eBooks.
def words(text): return re.findall(r'\w+', text.lower())

# Generates word frequency string form a Counter() object.
def count_dict_to_str(count_dict):
    return ''.join([word_set[0].decode('utf-8') + ' ' + str(word_set[1]) + '\n' \
        for word_set in count_dict.iteritems()])[:-1]

def update_dict(counter_dict, word, count):
    if not word.isdigit():
        if word not in counter_dict:
            counter_dict[word] = count
        else:
            counter_dict[word] += count


def main(argv):
    refresh = False
    parse_html = False
    clean = False

    try:
        opts, args = getopt.getopt(argv,'rc', ['refresh', 'parse-html', \
            'clean'])

    except getopt.GetoptError:
        print 'ERROR: Unknown parameter. Usage: generate_model.py' \
            + ' [-r] [-c] [--refresh], [--parse-html] [--clean]'
        sys.exit(2)

    for o, a in opts:
        if o == '-r' or o == '--refresh':
            refresh = True
        elif o == '--parse-html':
            parse_html = True
        elif o == '-c' or o == '--clean':
            refresh = True
            clean = True

    # If the info file does not exist, generates it.
    if not os.path.exists(MODEL_DIR + FILE_LIST_INFO_FILENAME):
        refresh = True

    start_time = datetime.now()

    if refresh:
        files = update_file_list_info(clean)
    else:
        # If the file already exists, loads it.
        files = ujson.load(open(MODEL_DIR + FILE_LIST_INFO_FILENAME, 'r'))

    file_text = []
    file_number = len(files)

    counter_dict = {}

    for file_index in trange(file_number, \
                desc='Generating word frequency dictionary', total=file_number):

        path = files[file_index]['path']
        word_frequency = files[file_index]['word_frequency']

        # If the word frequency file exist process it.
        if os.path.exists(word_frequency):
            with codecs.open(word_frequency, 'r', 'utf-8') as file:

                # Adds the news words or sum their count to the final model.
                for line in file.readlines():
                    word_count = line.rsplit(' ', 1)
                    
                    word = word_count[0]
                    count = int(word_count[1])

                    update_dict(counter_dict, word, count)

        else:
            # If the word frequency does not exist, it generates it.
            try:
                text = ''

                with codecs.open(path, 'r', 'utf-8') as file:

                    text = file.read() + '\n'

                    # Removes HTML tags form the eBook and overwrite the file.
                    if parse_html:
                        text = BeautifulSoup(text, 'html.parser')
                        try:
                            text = text.body.get_text()
                        except:
                            text = text.get_text()

                    count_dict = Counter(words(text))

                    # Adds the news words or sum their count to the final model.
                    for word_set in count_dict.iteritems():
                        
                        word = word_set[0]
                        value = word_set[1]

                        update_dict(counter_dict, word, count)

                    # Generates word frequency file from the eBook.
                    with codecs.open(word_frequency, 'w', 'utf-8') as wfout:
                        wfout.write(count_dict_to_str(count_dict))

                if parse_html:
                    with codecs.open(path, 'w', 'utf-8') as overwrite_f:
                        overwrite_f.write(text)

            except Exception, e:
                print e
                print type(e)
                print path
                # Renames the file name if it's too long.
                # os.rename(path, BOOK_DIR + str(uuid.uuid4()))

    concatenation_start_time = datetime.now()

    print 'Word frequency dictionary generation elapsed time: ' \
        + str(datetime.now() - concatenation_start_time)

    serialization_start_time = datetime.now()

    print 'Serializing word frequency dictionary.'

    # Saves the model into a file.
    with codecs.open(MODEL_DIR + OUTPUT_FILENAME, 'w', 'utf-8') as output:
        output.write(count_dict_to_str(counter_dict))

    print 'Serialization elapsed time: ' + str(datetime.now() \
        - serialization_start_time)

    print 'Total elapsed time: ' + str(datetime.now() - start_time)

if __name__ == '__main__':
    main(sys.argv[1:])
