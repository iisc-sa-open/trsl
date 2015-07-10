#! /usr/bin/env python2


class Node():

    def __init__(self):

        self.lchild = None
        self.rchild = None
        self.set = None
        self.distribution = None
        self.predictor_variable_index = None
        self.ngram_fragment_row_indices = None
        self.entropy = 0
        self.parent = None
