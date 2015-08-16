#! /usr/bin/env/python2
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Utilised for calculating reduction for each node
    Helps stores all the required attributes regarding
    the question asked in each node
"""

from collections import defaultdict

class Question(object):
    """
        Stores all the data that results from asking a question
        at a particular node
        todo : maybe a dict or named tuple will suffice

        Naming Convention:
            *    nb  -> Not belongs the the selected set
            *    b   -> Belongs to the selected set
    """

    def __init__(self):

        self.b_fragment = []
        self.nb_fragment = []
        self.b_dist = None
        self.nb_dist = None
        self.b_dist_entropy = 0
        self.nb_dist_entropy = 0
        self.reduction = 0
        self.set = set()
        self.predictor_variable_index = 0
        self.b_probability = 0
        self.nb_probability = 0
        self.avg_conditional_entropy = float("inf")
        # min of avg_conditional_entropy needs to be picked, thus default is set to infinity
