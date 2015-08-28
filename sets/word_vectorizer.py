#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Used to produce word vectors for the given file,
    using a pretrained word2vec model
"""

import sys
import gensim
import json


def generate_vectors(model=None, word_list=None, vector_file=None):
    """
        Used to generate vectors from pretrained word2vec model,
        for the entire word_list and write it to an output file
    """

    if word_list is None or model is None or vector_file is None:
        return None

    word2vec = gensim.models.word2vec.Word2Vec
    model = word2vec.load_word2vec_format(model, binary=True)

    for word in word_list:
        try:
            vector_file.write(json.dumps([word, model[word].tolist()]) + "\n")
        except KeyError:
            continue

if __name__ == "__main__":

    VECTOR_FILE = open(sys.argv[2], "w")
    WORD_LIST = open(sys.argv[1], "r").read().split()
    MODEL = sys.argv[3]
    generate_vectors(vector_file=VECTOR_FILE, model=MODEL, word_list=WORD_LIST)
