#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Preprocesses the data for trsl construction
"""

import copy
import sys

from nltk.tokenize import RegexpTokenizer, sent_tokenize

from ngram_table import NGramTable
reload(sys)
sys.setdefaultencoding('utf-8')


def preprocess(filename, ngram_window_size, sets):
    """
        Preprocesses the text present in the specified filename.
        * Tokenizes the data based on punctuation and spaces.
        * Constructs the ngram table based on the specified ngram window size.
        * Constructs a vocabulary set from the tokenized corpus.
        Return type:
            Tuple( ngram table, vocabulary set)
        Arguments:
            filename with corpus data, ngram window size, sets
    """

    corpus = open(filename, "r").read().encode(
        'ascii', errors='ignore').lower()
    # sentence tokenize the given corpus
    sentences = sent_tokenize(corpus)

    # word tokenize the given list of sentences
    tokenizer = RegexpTokenizer(r'(\w+(\'\w+)?)|\.')
    tokenized_corpus = [
        x for x in map(tokenizer.tokenize, sentences) if (
            len(x) >= ngram_window_size
        )
    ]
    word_ngram_table = NGramTable(tokenized_corpus, ngram_window_size)
    tokenized_corpus = copy.deepcopy(tokenized_corpus)
    # An index of which word belongs to which word set
    set_reverse_index = {}

    for i in xrange(len(sets)):
        for word in sets[i]:
            set_reverse_index[word] = i
    sets.append(set())
    for i in xrange(len(tokenized_corpus)):
        for j in xrange(len(tokenized_corpus[i])):
            try:
                tokenized_corpus[i][j] = set_reverse_index[
                    tokenized_corpus[i][j]]
            except KeyError:
                sets[-1].add(tokenized_corpus[i][j])
                set_reverse_index[tokenized_corpus[i][j]] = len(sets) - 1
                tokenized_corpus[i][j] = len(sets) - 1

    ngram_table = NGramTable(tokenized_corpus, ngram_window_size)
    return (ngram_table, sets, word_ngram_table)
