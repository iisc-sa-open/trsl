from nltk.tokenize import RegexpTokenizer
import numpy as np
from ngram_table import NGramTable


def preprocess(filename, ngram_window_size):
    """
        Preprocesses the text present in the specified filename.
        * Tokenizes the data based on punctuation and spaces.
        * Constructs the ngram table based on the specified ngram window size.
        * Constructs a vocabulary set from the tokenized corpus.
        Return type -> Tuple( ngram table, vocabulary set)
        Input arguments -> filename with corpus data, ngram window size.
    """

    corpus = open(filename, "r").read()
    tokenizer = RegexpTokenizer(r'\w+(\'\w+)?')
    tokenized_corpus = tokenizer.tokenize(corpus.lower())
    ngram_table = NGramTable(tokenized_corpus, ngram_window_size)
    vocabulary_set = set(tokenized_corpus)
    return (ngram_table, vocabulary_set)
