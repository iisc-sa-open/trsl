#! /usr/bin/env/python2
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Used to preprocess sets and sort them based on frequency
    and write them to a file
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from collections import Counter
from nltk.tokenize import RegexpTokenizer

def preprocess_corpus():
    """
        Preprocess corpus to obtain tokenized words
        sorted according to frequency
    """

    if len(sys.argv) > 1:
        for index in range(1, len(sys.argv)):
            filename = sys.argv[index]
            corpus = open(filename).read().encode('ascii', errors='ignore').lower()
            tokenizer = RegexpTokenizer(r'\w+(\'\w+)?')
            tokenized_corpus = tokenizer.tokenize(corpus.lower())
            output_file = open(filename+"-sorted", "w")
            freq_count = Counter(tokenized_corpus)
            for data in freq_count.most_common():
                word, count = data
                output_file.write(word+"\n")
            output_file.close()
    else:
        print "No Arguments Passed"

if __name__ == "__main__":

    preprocess_corpus()
