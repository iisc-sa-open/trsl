#! /usr/bin/env python2
"""
    NGramTable implemented
    todo : implement queries in ngram_table within the class itself
"""


class NGramTable(object):
    """
        Creates an abstraction over an
        array of objects to access them
        as if it were a stream of ngrams
        in table with rows and columns.
        Arguments:
            self.array             ->   Array of objects to access
                                        as an ngram
            self.ngram_window_size ->   The n in the ngram :D
    """

    def __init__(self, sentences, n):

        self.sentences = sentences
        self.ngram_window_size = n

    def generate_all_ngram_indices(self):

        for sentence_index in xrange(len(self.sentences)):
            for ngram_index in xrange(len(self.sentences[sentence_index]) - (self.ngram_window_size - 1)):
                yield (sentence_index, ngram_index)

    def __getitem__(self, tup):

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
