from nltk.tokenize import RegexpTokenizer
import numpy as np


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
    count = max(0, len(tokenized_corpus) - ngram_window_size + 1)
    ngram_table = np.array([list(tokenized_corpus[i:i+ngram_window_size]) for i in range(count)], np.object)
    vocabulary_set = set(tokenized_corpus)
    return (ngram_table, vocabulary_set)
