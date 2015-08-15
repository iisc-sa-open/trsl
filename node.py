#! /usr/bin/env/python2
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Node Class implemented
    which could in turn be used by another function to do the same
    with the entire decision tree.
"""

from question import Question
import math

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
        self.row_fragment_indices = []
        self.probabilistic_entropy = 0
        self.absolute_entropy = 0
        self.parent = None
        self.depth = 0
        self.probability = 0
        self.best_question = None
        self.set_known_predvars = [False for x in xrange(ngram_window_size - 1)]

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
        # Iterate over the path traversed for reaching the node ensuring question is unique
        while parent is not None:
            if parent.set == set_index and parent.predictor_variable_index == x_index:
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
                # We already know what set this predictor variable belongs to in
                # this node's slice of data. So no point asking this question
                # The reduction is set to 0 by default for a question
                yield question

            question.set = set_index
            question.predictor_variable_index = x_index
            self.count_target_word_frequencies(ngram_table, x_index, set_index, question)
            question.b_dist_entropy = self.frequencies_to_probabilities_and_entropy(question.b_dist)
            question.nb_dist_entropy = self.frequencies_to_probabilities_and_entropy(question.nb_dist)

            size_row_fragment = (
                len(self.row_fragment_indices)
            )
            # Probability for next node in YES path computed
            question.b_probability = 0 if size_row_fragment is 0 else (
                self.probability * float(len(question.b_indices))/size_row_fragment
            )
            # Probability for next node in No path computed
            question.nb_probability = 0 if size_row_fragment is 0 else (
                self.probability * float(len(question.nb_indices))/size_row_fragment
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


    def count_target_word_frequencies(self, ngram_table, x_index, set_index, question):
        """
            Count target word frequencies for predictor
            variable belongs to set and predictor variable
            does not belong to set
            Argument:
                ngram_table -> Instance of NgramTable
                x_index     -> Predictor Variable index
                set_index   -> Set index
                question    -> Question asked
        """

        for sentence_index, ngram_index in self.row_fragment_indices:
            predictor_word = ngram_table[sentence_index, ngram_index, x_index]
            target_word = ngram_table[
                sentence_index, ngram_index, ngram_table.ngram_window_size-1
            ]
            # Target word belongs to the set
            if predictor_word == set_index:
                question.b_indices.append((sentence_index, ngram_index))
                question.b_dist[target_word] += 1.0
            # Target word does not belong to the set
            else:
                question.nb_indices.append((sentence_index, ngram_index))
                question.nb_dist[target_word] += 1.0

    def frequencies_to_probabilities_and_entropy(self, frequency_map):
        """
            Compute probability from frequency of occurence
            and return the entropy of the same from the
            frequency_map entries
            Argument:
                frequency_map -> {keys, values}
                    keys    -> Set_id
                    values  -> count of target variable
            Return:
                entropy
        """

        frequency_sum = sum(frequency_map.values())
        entropy = 0
        # Iterate over the frequency_map computing the entropy, probability
        for key in frequency_map.keys():
            frequency = frequency_map[key]
            probability = frequency / frequency_sum
            probability_of_info_gain = (
                probability * math.log(probability, 2)
            )
            frequency_map[key] = probability
            entropy += -probability_of_info_gain

        return entropy

    def is_leaf(self):
        """
            Checks if the current node is a leafnode or an internal node
            by checking if it has any children or not
        """

        # Check any one leaf node suffices
        if self.rchild is not None or self.lchild is not None:
            return False
        else:
            return True
