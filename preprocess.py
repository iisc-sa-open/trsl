"""
    Preprocesses the data for trsl construction
"""

from nltk.tokenize import RegexpTokenizer
from ngram_table import NGramTable


def preprocess(filename, ngram_window_size, sets):
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
    word_ngram_table = NGramTable(tokenized_corpus, ngram_window_size)
    set_reverse_index = {}
    
    # Make a copy of the tokens, so we can just deal with set indices
    # henceforth. The actual words are needed again to compute word
    # probabilities at leaf nodes after the tree has stopped growing
    tokenized_corpus = list(tokenized_corpus)
    for i in xrange(len(sets)):
        for word in sets[i]:
            set_reverse_index[word] = i
    sets.append([])
    for i in xrange(len(tokenized_corpus)):
        try:
            tokenized_corpus[i] = set_reverse_index[tokenized_corpus[i]]
        except KeyError:
            sets[- 1].append(tokenized_corpus[i])
            set_reverse_index[tokenized_corpus[i]] = len(sets) - 1
            tokenized_corpus[i] = len(sets) - 1
    ngram_table = NGramTable(tokenized_corpus, ngram_window_size)
    return (ngram_table, word_ngram_table, sets, set_reverse_index)
