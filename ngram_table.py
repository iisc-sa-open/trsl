#! /usr/bin/env/python2
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    NGramTable Abstraction implemented which produces a moving ngram_window_size
    over individual sentences in the corpus
"""


class NGramTable(object):
    """
        Creates an abstraction over an array of sentences to access them
        as if it were a stream of ngrams in table with rows and columns.
    """

    def __init__(self, sentences, n):
        """
            Constructor for instantiating an abstraction of NGramTable
            which produces moving ngram_window_size over the sentences
            Arguments:
                self.sentences         ->   Array of sentences
                self.ngram_window_size ->   The n in the ngram :D
        """

        self.sentences = sentences
        self.ngram_window_size = n

    def generate_all_ngram_indices(self):
        """
            Generator for producing ngram window for each individual sentences
            in the corpus.
            Return type:
                sentence_index   ->  sentence index
                ngram_index      ->  ngram index in that sentence
        """

        for sentence_index in xrange(len(self.sentences)):
            for ngram_index in xrange(len(self.sentences[sentence_index]) - (self.ngram_window_size - 1)):
                yield (sentence_index, ngram_index)

    def __getitem__(self, tup):
        """
            Returns the word in the ngram window for a specific sentence
            Argument:
                tuple -> sentence_id, ngram_id, word_id
            Return type:
                word  -> at sentence_id, ngram_id, word offset
        """

        sentence_index, ngram_index, word_index = tup

        if sentence_index < len(self.sentences):
            if ngram_index < ( len(self.sentences[sentence_index]) - (self.ngram_window_size - 1) ):
                if word_index < self.ngram_window_size:
                    return self.sentences[sentence_index][ngram_index + word_index]
                else:
                    raise KeyError('Word index out of range')
            else:
                raise KeyError('Ngram index out of range')
        else:
            raise KeyError('Sentence index out of range')
