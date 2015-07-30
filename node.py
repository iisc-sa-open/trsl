#! /usr/bin/env python2
"""
    Node Class implemented
    todo: Add functionality to serialize and load nodes,
    which could in turn be used by another function to do the same
    with the entire deciset_dataon tree.
"""
from question import Question
import math

class Node(object):
    """
        A class which holds all the attributes of a node in the
        deciset_dataon tree

        todo: Evaluate if a named tuple or dictionary would suffice
    """

    def __init__(self):

        self.lchild = None
        self.rchild = None
        self.set = None
        self.dist = None
        self.predictor_variable_index = None
        self.row_fragment_indices = None
        self.probabilistic_entropy = 0
        self.parent = None
        self.depth = 0
        self.probability = 0

    def question_already_asked(self, x_index, set_data):
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
        while parent is not None:
            if parent.set == set_data and parent.predictor_variable_index == x_index:
                return True
            else:
                parent = parent.parent
        return False

    def eval_question(self, ngram_table, pred_var_set_pair):
        """
            Evaluate question by computing the avg conditional entropy,
            reduction, belongs to and not belongs to probability
        """

        x_index, set_data = pred_var_set_pair
        question = Question()
        if self.question_already_asked(x_index, set_data):
            #The reduction is set to 0 by default for a question
            return question

        question.set = set_data
        question.predictor_variable_index = x_index
        self.count_target_word_frequencies(ngram_table, x_index, set_data, question)
        question.b_dist_entropy = self.frequencies_to_probabilities_and_entropy(question.b_dist)
        question.nb_dist_entropy = self.frequencies_to_probabilities_and_entropy(question.nb_dist)

        set_dataze_row_fragment = (
            len(self.row_fragment_indices)
        )
        question.b_probability = (
            self.probability * float(len(question.b_indices))/set_dataze_row_fragment
        )
        question.nb_probability = (
            self.probability * float(len(question.nb_indices))/set_dataze_row_fragment
        )
        question.avg_conditional_entropy = (
            (question.b_probability * question.b_dist_entropy)
            +
            (question.nb_probability * question.nb_dist_entropy)
        )
        question.reduction = (
            self.probabilistic_entropy - question.avg_conditional_entropy
        )

        return question


    def count_target_word_frequencies(self, ngram_table, x_index, set_data, question):
        """
            Count target word frequencies for predictor
            variable belongs to set and predictor variable
            does not belong to set
        """

        for table_index in self.row_fragment_indices:
            predictor_word = ngram_table[table_index, x_index]
            target_word = ngram_table[
                table_index, ngram_table.ngram_window_size-1
            ]
            if predictor_word in set_data:
                question.b_indices.append(table_index)
                try:
                    question.b_dist[target_word] += 1.0
                except KeyError:
                    question.b_dist[target_word] = 1.0
            else:
                question.nb_indices.append(table_index)
                try:
                    question.nb_dist[target_word] += 1.0
                except KeyError:
                    question.nb_dist[target_word] = 1.0

    def frequencies_to_probabilities_and_entropy(self, hashmap):
        """
            Compute probability from frequency of occurence
            and return the entropy of the same from the
            hasmap entries
        """

        frequency_sum = sum(hashmap.values())
        entropy = 0
        for key in hashmap.keys():
            frequency = hashmap[key]
            probability = frequency / frequency_sum
            probability_of_info_gain = (
                probability * math.log(probability, 2)
            )
            hashmap[key] = probability
            entropy += -probability_of_info_gain

        return entropy
