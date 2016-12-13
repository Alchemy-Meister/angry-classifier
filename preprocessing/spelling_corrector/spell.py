# /usr/bin/env
# -*- coding: utf-8 -*-

"""Spelling Corrector in Python 3; see http://norvig.com/spell-correct.html

Copyright (c) 2007-2016 Peter Norvig
MIT license: www.opensource.org/licenses/mit-license.php
"""

################ Spelling Corrector 

import os
import re
import codecs
from collections import Counter
from math import log10


def words(text): return re.findall(r'\w+', text.lower())

script_dir = os.path.dirname(__file__)
model_rel_path = 'language_model/english (wf)'
model_abs_path = os.path.join(script_dir, model_rel_path)

WORDS = Counter()

with codecs.open(model_abs_path, 'r', 'utf-8') as file:

    for line in file.readlines():
        word_count = line.rsplit(' ', 1)
        
        word = word_count[0]
        count = int(word_count[1])

        if word not in WORDS:
            WORDS[word] = count
        else:
            WORDS[word] += count

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) \
        or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts    = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
