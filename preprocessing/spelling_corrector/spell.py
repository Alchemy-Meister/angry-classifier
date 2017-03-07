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

WORDS = Counter()

# Filter to remove words with an appearance lower than defined.
FILTER_NUMBER = 4000

script_dir = os.path.dirname(__file__)
model_rel_path = 'language_model/english (wf)'
model_abs_path = os.path.join(script_dir, model_rel_path)

with codecs.open(model_abs_path, 'r', 'utf-8') as file:

    for line in file.readlines():
        word_count = line.rsplit(' ', 1)
        
        try:
            word = str(word_count[0])
        except:
            word = word_count[0]
        count = int(word_count[1])

        if count >= FILTER_NUMBER:
            if word not in WORDS:
                WORDS[word] = count
            else:
                WORDS[word] += count

def words(text): return re.findall(r'\w+', text.lower())

# Uncomment to use Norving's Language Model.
#WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return float(WORDS[word]) / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return set(known([word]) or known(edits1(word)) or known(edits2(word)) \
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

################ Test Code 

def unit_tests():
    print correction('speling')        # insert
    print correction('korrectud')        # replace 2
    print correction('bycycle')        # replace
    print correction('inconvient')       # insert 2
    return 'unit_tests pass'

def spelltest(tests, verbose=False):
    "Run correction(wrong) on all (right, wrong) pairs; report results."
    import time
    start = time.clock()
    good, unknown = 0, 0
    n = len(tests)
    for right, wrong in tests:
        w = correction(wrong)
        good += int(w == right)
        if w != right:
            found = False
            for currentWord in WORDS:
                if currentWord == right:
                    found = True
            unknown += int(not found)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'
                      .format(wrong, w, WORDS[w], right, WORDS[right]))
    dt = time.clock() - start
    print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
          .format(float(good) / n, n, float(unknown) / n, float(n) / dt))
    
def Testset(lines):
    "Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."
    return [(right, wrong)
            for (right, wrongs) in (line.split(':') for line in lines)
            for wrong in wrongs.split()]

if __name__ == '__main__':
    print(unit_tests())
    spelltest(Testset(open('spell-testset1.txt')))
    spelltest(Testset(open('spell-testset2.txt')))