#! /usr/bin/env python2


class Node():

    def __init__(self):

        self.lchild = None
        self.rchild = None
        self.set = None
        self.dist = None
        self.predictor_variable_index = None
        self.row_fragment_indices = None
        self.entropy = 0
        self.parent = None
