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

    def __init__(self, arr, n):

        self.array = arr
        self.ngram_window_size = n

    def __len__(self):

        return len(self.array) - (self.ngram_window_size - 1)

    def __getitem__(self, tup):

        row, col = tup
        if row < (len(self.array) - (self.ngram_window_size - 1)) and col < self.ngram_window_size:
            return self.array[row + col]
        else:
            raise KeyError
