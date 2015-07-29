#! /usr/bin/env python2
"""
    Utilised for calculating reduction
"""


class Question(object):
    """
        Stores all the data that results from asking a question
        at a particular node
        todo : maybe a dict or named tuple will suffice
    """

    def __init__(self):

        self.b_indices = []
        self.nb_indices = []
        self.b_dist = {}
        self.nb_dist = {}
        self.b_dist_entropy = 0
        self.nb_dist_entropy = 0
        self.reduction = 0
        self.set = set()
        self.predictor_variable_index = 0
        self.b_probability = 0
        self.nb_probability = 0
        self.avg_conditional_entropy = float("inf")
        # min of avg_conditional_entropy needs to be picked, thus default is set to infinity
