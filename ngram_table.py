#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    NGramTable Abstraction implemented which
    produces a moving ngram_window_size
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

    def generate_all_ngrams(self):
        """
            Generator for producing ngram window for each individual sentences
            in the corpus.
            Return type:
                An array of ngram_window_size items
        """

        for sentence_index in xrange(len(self.sentences)):
            no_of_ngrams = (
                len(
                    self.sentences[sentence_index]
                )-(self.ngram_window_size - 1)
            )
            for ngram_index in xrange(no_of_ngrams):
                yield self.sentences[sentence_index][
                    ngram_index:ngram_index + self.ngram_window_size
                ]

    def __getitem__(self, tup):
        """
            Returns the word in the ngram window for a specific sentence
            Argument:
                tuple -> sentence_id, ngram_id, word_id
            Return type:
                word  -> at sentence_id, ngram_id, word offset
        """

        sentence_index, ngram_index, word_index = tup
        return self.sentences[sentence_index][ngram_index + word_index]
