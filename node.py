#! /usr/bin/env python2
"""
    Node Class implemented
    todo: Add functionality to serialize and load nodes,
    which could in turn be used by another function to do the same
    with the entire decision tree.
"""


class Node(object):
    """
        A class which holds all the attributes of a node in the
        decision tree

        todo: Evaluate if a named tuple or dictionary would suffice
    """

    def __init__(self):

        self.lchild = None
        self.rchild = None
        self.set = None
        self.dist = None
        self.predictor_variable_index = None
        self.row_fragment_indices = None
        self.entropy = 0
        self.parent = None
        self.depth = 0
