#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Node Class implemented
    which could in turn be used by another function to do the same
    with the entire decision tree.
"""

from question import Question
import scipy.stats
import numpy as np


class Node(object):

    """
        A class which holds all the attributes of a node in the
        decision tree

        todo: Evaluate if a named tuple or dictionary would suffice
    """

    def __init__(self, ngram_window_size):

        self.lchild = None
        self.rchild = None
        self.set = None
        self.dist = None
        self.predictor_variable_index = None
        self.data_fragment = None
        self.probabilistic_entropy = 0
        self.absolute_entropy = 0
        self.parent = None
        self.depth = 0
        self.probability = 0
        self.best_question = None
        self.len_data_fragment = 0
        self.word_probability = None
        self.set_known_predvars = (
            [False for x in xrange(ngram_window_size - 1)]
        )

    def question_already_asked(self, x_index, set_index):
        """
            Checks if the same question has been asked
            (same set and predictor variable index)
            in the parent and the parent's parent and so on
            till the root.

            The rationale here is that asking the same question again
            on a subset of the data the question was asked before
            would cause all the data to go down one one of YES or NO path
            which is unnecessary computation.
        """

        parent = self.parent
        # Iterate over the path traversed for ensuring question is unique
        while parent is not None:
            question_index_asked = (
                parent.predictor_variable_index == x_index
            )
            if (parent.set == set_index) and (question_index_asked):
                return True
            else:
                parent = parent.parent
        return False

    def generate_questions(self, ngram_table, pred_var_set_pairs_generator):
        """
            Evaluate question by computing the avg conditional entropy,
            reduction, belongs to and not belongs to probability
        """

        for pred_var_set_pair in pred_var_set_pairs_generator():
            x_index, set_index = pred_var_set_pair
            question = Question()
            if self.question_already_asked(x_index, set_index):
                # The reduction is set to 0 by default for a question
                yield question

            if self.set_known_predvars[x_index]:
                # We know what set this predictor variable belongs to in
                # this node's slice of data. So no point asking this question
                # The reduction is set to 0 by default for a question
                yield question

            question.set = set_index
            question.predictor_variable_index = x_index
            condition = self.data_fragment[:, x_index] == set_index
            question.b_fragment = self.data_fragment.compress(
                condition, axis=0)
            question.nb_fragment = (
                self.data_fragment.compress(~condition, axis=0)
            )

            target_column_index = self.data_fragment.shape[1] - 1
            b_probabilities = np.bincount(
                question.b_fragment[:, target_column_index]
            ).astype('float32') / question.b_fragment.shape[0]
            nb_probabilities = np.bincount(
                question.nb_fragment[:, target_column_index]
            ).astype('float32') / question.nb_fragment.shape[0]

            question.b_dist = {
                index: b_probabilities[index] for index in range(
                    len(b_probabilities)
                )
            }
            question.nb_dist = {
                index: nb_probabilities[index] for index in range(
                    len(nb_probabilities)
                )
            }

            question.b_dist_entropy = scipy.stats.entropy(
                b_probabilities, base=2
            )
            question.nb_dist_entropy = scipy.stats.entropy(
                nb_probabilities, base=2
            )

            size_data = (
                self.data_fragment.shape[0]
            )
            # Probability for next node in YES path computed
            question.b_probability = 0 if size_data is 0 else (
                self.probability * float(
                    question.b_fragment.shape[0]
                ) / size_data
            )
            # Probability for next node in No path computed
            question.nb_probability = 0 if size_data is 0 else (
                self.probability * float(
                    question.nb_fragment.shape[0]
                ) / size_data
            )
            # Avg conditional entropy computed for the node
            question.avg_conditional_entropy = (
                (question.b_probability * question.b_dist_entropy)
                +
                (question.nb_probability * question.nb_dist_entropy)
            )
            # Reduction computed for current node
            question.reduction = (
                self.probabilistic_entropy - question.avg_conditional_entropy
            )

            yield question

    def is_leaf(self):
        """
            Checks if the current node is a leafnode or an internal node
            by checking if it has any children or not
        """

        # Check any one leaf node suffices
        return True if self.rchild is None else False
