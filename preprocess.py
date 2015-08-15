#! /usr/bin/env/python2
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Preprocesses the data for trsl construction
"""

from nltk.tokenize import RegexpTokenizer, sent_tokenize
from ngram_table import NGramTable
import copy

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

    corpus = open(filename, "r").read().lower()
    # sentence tokenize the given corpus
    sentences = sent_tokenize(corpus)
    # word tokenize the given list of sentences
    tokenizer = RegexpTokenizer(r'(\w+(\'\w+)?)|\.')
    tokenized_corpus = filter(
        lambda x: len(x) >= ngram_window_size,
        map(tokenizer.tokenize, sentences)
    )
    # instantiate NGramTable for the given corpus based on ngram window size
    word_ngram_table = NGramTable(tokenized_corpus, ngram_window_size)
    set_reverse_index = {}

    # Make a copy of the tokens, so we can just deal with set indices
    # henceforth. The actual words are needed again to compute word
    # probabilities at leaf nodes after the tree has stopped growing
    tokenized_corpus = copy.deepcopy(tokenized_corpus)
    for i in xrange(len(sets)):
        for word in sets[i]:
            set_reverse_index[word] = i
    sets.append([])
    for i in xrange(len(tokenized_corpus)):
        for j in xrange(len(tokenized_corpus[i])):
            try:
                tokenized_corpus[i][j] = set_reverse_index[tokenized_corpus[i][j]]
            except KeyError:
                sets[-1].append(tokenized_corpus[i][j])
                set_reverse_index[tokenized_corpus[i][j]] = len(sets) - 1
                tokenized_corpus[i][j] = len(sets) - 1
    ngram_table = NGramTable(tokenized_corpus, ngram_window_size)
    return (ngram_table, word_ngram_table, sets, set_reverse_index)
