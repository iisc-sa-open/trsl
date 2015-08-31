#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Used to preprocess sets and sort them based on frequency
    and write them to a file
"""

from collections import Counter
import sys

from nltk.tokenize import RegexpTokenizer
reload(sys)
sys.setdefaultencoding('utf-8')


def preprocess_corpus():
    """
        Preprocess corpus to obtain tokenized words
        sorted according to frequency
    """

    if len(sys.argv) > 1:
        for index in range(1, len(sys.argv)):
            filename = sys.argv[index]
            corpus = open(filename).read().encode(
                'ascii',
                errors='ignore'
            ).lower()
            tokenizer = RegexpTokenizer(r'\w+(\'\w+)?')
            tokenized_corpus = tokenizer.tokenize(corpus.lower())
            output_file = open(filename + "-sorted", "w")
            freq_count = Counter(tokenized_corpus)
            for data in freq_count.most_common():
                word, _ = data  # word, count
                output_file.write(word + "\n")
            output_file.close()
    else:
        print "No Arguments Passed"

if __name__ == "__main__":

    preprocess_corpus()
