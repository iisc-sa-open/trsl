#! /usr/bin/env/python2
"""
    Tree Based Statistical language Model
    to be used in tandem with an acoustic model
    for speech recognition
"""


from collections import Counter   # todo : remove this
from node import Node
from question import Question
import json
import logging
import math
import preprocess   # todo check for namespace pollution
import Queue
import random
import pickle


class Trsl(object):
    """
        Trsl class implements a tree based statistical language model
        Arguments:
            self.reduction_threshold -> min reduction in entropy for a question
                                        at a node to further grow the tree
            self.ngram_window_size   -> no of predictor variables (inclusive of target)
    """

    def __init__(self, filename, ngram_window_size=5, reduction_threshold=1):

        self.reduction_threshold = reduction_threshold
        self.ngram_window_size = ngram_window_size
        self.root = Node()
        self.previous_best_questions = set()
        self.no_of_nodes = 1
        self.ngram_table = None
        self.vocabulary_set = None
        self.word_sets = None
        # inf queue size hopefully <- todo : evaluate if we need this
        self.node_queue = Queue.Queue(-1)
        self.max_depth = 0
        self.min_depth = float('inf')
        self.filename = filename

    def train(self):
        """
            Given a filename(containing a training corpus), build a decision tree
            which can be accessed through self.root

        """

        try:
            open(self.filename + ".dat", "r")
            logging.info("Found dat file -> loading precomputed data")
            self.load(self.filename + ".dat")
        except (OSError, IOError) as e:
            self.ngram_table, self.vocabulary_set = preprocess.preprocess(
                self.filename, self.ngram_window_size
            )
            self.set_root_state()
            self.node_queue.put(self.root)
            self.word_sets = self.build_sets()
            while not self.node_queue.empty():
                self.process_node(self.node_queue.get())
            logging.info("Total no of Nodes:"+ str(self.no_of_nodes))
            logging.info(
                    "Max Depth: %s Min Depth: %s"
                    % (
                        self.max_depth,
                        self.min_depth
                    )
                )
            self.serialize(self.filename + ".dat")

    def set_root_state(self):
        """
            Calculates the probability distribution
            of the target variable values at the root node
            in the training text and also
            the entropy of this distribution.
            Also, adds all the row indices in the ngram table
            into row_fragment_indices since the data is not
            fragmented yet.
        """
        self.root.dist = {}
        for index in range(0, len(self.ngram_table)):
            try:
                self.root.dist[
                    self.ngram_table[index, self.ngram_window_size-1]
                    ] += 1.0
            except KeyError:
                self.root.dist[
                    self.ngram_table[index, self.ngram_window_size-1]
                    ] = 1.0

        # todo check if computes through entrie ds or not
        self.root.row_fragment_indices = [
            x for x in range(0, len(self.ngram_table))
        ]

        for key in self.root.dist.keys():
            frequency = self.root.dist[key]
            probability = frequency/len(self.root.row_fragment_indices)
            probability_of_info_gain = probability * math.log(probability, 2)
            self.root.dist[key] = probability
            self.root.entropy += -probability_of_info_gain

        logging.debug(
            "Root Entropy: %s"
            % (
                self.root.entropy
            )
        )

    def process_node(self, curr_node):
        """
            Used to process curr_node by computing best reduction
            and choosing to create children nodes or not based on
            self.reduction_threshold

            Naming Convention:
                *    nb  -> Not belongs the the selected set
                *    b   -> Belongs to the selected set
                *    Xi  -> Predictor variable index into the ngram table
                *    Si  -> Set
                *    cnb -> current node not belongs to the selected set
                *    cb  -> current node belongs to the set
        """

        best_question = Question()
        self.no_of_nodes += 1
        for Xi in range(0, self.ngram_window_size-1):
            for Si in self.word_sets:
                if self.question_already_asked(curr_node, Xi, Si):
                    continue
                curr_question = Question()
                curr_question.set = Si
                curr_question.predictor_variable_index = Xi
                for table_index in curr_node.row_fragment_indices:
                    predictor_word = self.ngram_table[table_index, Xi]
                    target_word = self.ngram_table[
                        table_index, self.ngram_window_size-1
                    ]
                    if predictor_word in Si:
                        curr_question.b_indices.append(table_index)
                        try:
                            curr_question.b_dist[target_word] += 1.0
                        except KeyError:
                            curr_question.b_dist[target_word] = 1.0
                    else:
                        curr_question.nb_indices.append(table_index)
                        try:
                            curr_question.nb_dist[target_word] += 1.0
                        except KeyError:
                            curr_question.nb_dist[target_word] = 1.0
                b_frequency_sum = sum(curr_question.b_dist.values())
                for key in curr_question.b_dist.keys():
                    frequency = curr_question.b_dist[key]
                    probability = frequency/b_frequency_sum
                    probability_of_info_gain = (
                        probability * math.log(probability, 2)
                    )
                    curr_question.b_dist[key] = probability
                    curr_question.b_dist_entropy += -probability_of_info_gain

                nb_frequency_sum = sum(curr_question.nb_dist.values())
                for key in curr_question.nb_dist.keys():
                    frequency = curr_question.nb_dist[key]
                    probability = frequency/nb_frequency_sum
                    probability_of_info_gain = (
                        probability * math.log(probability, 2)
                    )
                    curr_question.nb_dist[key] = probability
                    curr_question.nb_dist_entropy += -probability_of_info_gain
                size_row_fragment = (
                    len(curr_node.row_fragment_indices)
                )
                curr_question.b_probability = (
                    float(len(curr_question.b_indices))/size_row_fragment
                )
                curr_question.nb_probability = (
                    float(len(curr_question.nb_indices))/size_row_fragment
                )
                curr_question.avg_conditional_entropy = (
                    (curr_question.b_probability
                        * curr_question.b_dist_entropy)
                    +
                    (curr_question.nb_probability
                        * curr_question.nb_dist_entropy)
                )
                curr_question.reduction = (
                    curr_node.entropy - curr_question.avg_conditional_entropy
                )
                if best_question.reduction < curr_question.reduction:
                    best_question = curr_question

        if best_question.reduction * 100 / curr_node.entropy > self.reduction_threshold:
            logging.debug(
                "Best Question: Reduction: %s -> X%s for Set: %s"
                % (
                    best_question.reduction,
                    best_question.predictor_variable_index,
                    best_question.set, 
                )
            )
            curr_node.set = best_question.set
            curr_node.predictor_variable_index = (
                best_question.predictor_variable_index
            )
            curr_node.lchild = Node()
            curr_node.lchild.row_fragment_indices = best_question.b_indices
            curr_node.lchild.entropy = best_question.b_dist_entropy
            curr_node.lchild.dist = best_question.b_dist
            curr_node.lchild.depth = curr_node.depth + 1
            curr_node.rchild = Node()
            curr_node.rchild.row_fragment_indices = best_question.nb_indices
            curr_node.rchild.entropy = best_question.nb_dist_entropy
            curr_node.rchild.dist = best_question.nb_dist
            curr_node.rchild.depth = curr_node.depth + 1
            if curr_node.lchild.entropy > 0:
                self.node_queue.put(curr_node.lchild)
            else:
                if curr_node.depth + 1 > self.max_depth:
                    self.max_depth = curr_node.depth + 1
                if curr_node.depth + 1 < self.min_depth:
                    self.min_depth = curr_node.depth - 1
                logging.info(
                    "Leaf Node reached, Top Probability dist:%s",
                    dict(Counter(curr_node.lchild.dist).most_common(5))
                )
            if curr_node.rchild.entropy > 0:
                self.node_queue.put(curr_node.rchild)
            else:
                if curr_node.depth + 1 > self.max_depth:
                    self.max_depth = curr_node.depth + 1
                if curr_node.depth + 1 < self.min_depth:
                    self.min_depth = curr_node.depth - 1
                logging.info(
                    "Leaf Node reached, Top Probability dist:%s",
                    dict(Counter(curr_node.rchild.dist).most_common(5))
                )
            curr_node.lchild.parent = curr_node
            curr_node.rchild.parent = curr_node

        else:
            if curr_node.depth + 1 > self.max_depth:
                self.max_depth = curr_node.depth + 1
            if curr_node.depth + 1 < self.min_depth:
                self.min_depth = curr_node.depth - 1
            logging.info(
                "Leaf Node reached, Top Probability dist:%s",
                dict(Counter(curr_node.dist).most_common(5))
            )

    def question_already_asked(self, curr_node, Xi, Si):
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
        parent = curr_node.parent
        while parent is not None:
            if parent.set == Si and parent.predictor_variable_index == Xi:
                return True
            else:
                parent = parent.parent
        return False

    def build_sets(self):

        """
            This method is stubbed right now to return predetermined sets
            We hope to build a system that returns sets built by clustering words
            based on their semantic similarity and semantic relatedness in the given
            training corpus.

            todo : make no of sets and their size configurable if possible
        """

        data = json.loads(open("../sets/Kmeans-9811words-100clusters.json","r").read())
 
        # Every element in the list needs to be a set because
        # belongs to operation utilises O(1) steps

        for i in range(len(data)):
            data[i] = set(data[i])
        return data

    def serialize(self, filename):
        open(filename,"wb").write(pickle.dumps(self.root))

    def load(self, filename):
        self.root = pickle.loads(open(filename,"rb").read())

    def tree_walk(self, seed, no_of_words):

        for x in range(no_of_words):
            dist = self.predict(seed[-(self.ngram_window_size-1)::])
            r = random.random()
            sum = 0
            for i in dist.keys():
                sum += dist[i]
                if r <= sum:
                    seed.append(i)
                    break
        return seed

    def predict(self, predictor_variable_list):
        """
            Given a list of predictor words, this method
            returns the probability distribution of the
            next words that could occur next.

            todo: exception handling when predict is called before train
        """

        if len(predictor_variable_list) != self.ngram_window_size-1:
            raise ValueError(
                "predictor_variable_list size should conform with ngram window size"
            )
        temp = self.root
        steps = 0
        while True:
            if temp.rchild is not None:
                # since the decision tree is  a full binary tree, both children exist
                steps += 1
                if predictor_variable_list[temp.predictor_variable_index] in temp.set:
                    logging.debug(
                        "LEVEL: %s, X%s = %s belongs to %s? YES"
                        % (
                            steps, temp.predictor_variable_index,
                            predictor_variable_list[
                                temp.predictor_variable_index
                            ],
                            temp.set
                        )
                    )
                    temp = temp.lchild
                else:
                    logging.debug(
                        "LEVEL: %s, X%s = %s belongs to %s? NO"
                        % (
                            steps, temp.predictor_variable_index,
                            predictor_variable_list[
                                temp.predictor_variable_index
                            ],
                            temp.set
                        )
                    )
                    temp = temp.rchild
            else:
                logging.info(
                    "Total Reduction in Entropy: %s -> %s%%"
                    % (
                        self.root.entropy - temp.entropy,
                        100 * (self.root.entropy - temp.entropy)/self.root.entropy
                    )
                )
                logging.debug(
                    "Probable Distribution: " + str(
                        temp.dist
                    )
                )
                logging.info("Depth Reached: " + str(
                        steps
                    )
                )
                return temp.dist
