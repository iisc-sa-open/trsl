#! /usr/bin/env/python2
"""
    Tree Based Statistical language Model
    to be used in tandem with an acoustic model
    for speech recognition
"""


from collections import Counter
from functools import partial
from node import Node
import json
import logging
import math
import preprocess   # todo check for namespace pollution
import Queue
import random
import pickle
import ConfigParser


class Trsl(object):
    """
        Trsl class implements a tree based statistical language model
        Arguments:
            self.reduction_threshold -> min reduction in entropy for a question
                                        at a node to further grow the tree
            self.ngram_window_size   -> no of predictor variables (inclusive of target)
    """

    def __init__(
            self,
            filename=None,
            ngram_window_size=None,
            reduction_threshold=None,
            set_filename=None
        ):

        if ((filename is None or ngram_window_size is None) or
                (reduction_threshold is None or set_filename is None)):
            config = ConfigParser.RawConfigParser()
            if len(config.read('config.cfg')) > 0:
                try:
                    if reduction_threshold is None:
                        reduction_threshold = config.getfloat('Trsl', 'reduction_threshold')
                    if filename is None:
                        filename = config.get('Trsl', 'corpus_filename')
                    if ngram_window_size is None:
                        ngram_window_size = config.getint('Trsl', 'ngram_window_size')
                    if set_filename is None:
                        set_filename = config.get("Trsl", 'set_filename')
                except ValueError:
                    logging.error("Error! ValueError occured in specified configuration")
                    raise ValueError
            else:
                logging.error("Config File not found: config.cfg")
                raise IOError
        logging.info("Reduction Threshold: %s", reduction_threshold)
        logging.info("Corpus Filename: %s", filename)
        logging.info("Ngram Window Size: %s", ngram_window_size)
        logging.info("Set Utilised: %s", set_filename)
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
        self.set_filename = set_filename

    def train(self):
        """
            Given a filename(containing a training corpus), build a decision tree
            which can be accessed through self.root

        """

        try:
            open(self.filename + ".dat", "r")
            logging.info("Found dat file -> loading precomputed data")
            self.__load(self.filename + ".dat")
            # todo: configuration not loaded from the file, reset parameters
        except (OSError, IOError):
            self.ngram_table, self.vocabulary_set = preprocess.preprocess(
                self.filename, self.ngram_window_size
            )
            self.__set_root_state()
            self.node_queue.put(self.root)
            self.word_sets = self.__build_sets()
            while not self.node_queue.empty():
                self.__process_node(self.node_queue.get())
            logging.info("Total no of Nodes:"+ str(self.no_of_nodes))
            logging.info(
                "Max Depth: %s Min Depth: %s",
                self.max_depth,
                self.min_depth
            )
            self.__serialize(self.filename + ".dat")

    def __set_root_state(self):
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
            "Root Entropy: %s",
            self.root.entropy
        )

    def __generate_pred_var_set_pairs(self):
        """
            Lazy generator for Predictor, Set combinations
        """

        for x_index in range(0, self.ngram_window_size-1):
            for set_data in self.word_sets:
                yield (x_index, set_data)

    def __process_node(self, curr_node):
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


        self.no_of_nodes += 1
        #bind ngramtable to a partial function
        eval_question = partial(curr_node.eval_question, self.ngram_table)
        questions = map(eval_question, self.__generate_pred_var_set_pairs())

        best_question = min(questions, key=lambda question: question.avg_conditional_entropy)
        if best_question.reduction * 100 / curr_node.entropy > 1:
            logging.debug(
                "Best Question: Reduction: %s -> X%s for Set: %s",
                best_question.reduction,
                best_question.predictor_variable_index,
                id(best_question.set)
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
            if curr_node.lchild.entropy > 0 and ((self.root.entropy - curr_node.lchild.entropy) * 100 / self.root.entropy < self.reduction_threshold):
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
            if curr_node.rchild.entropy > 0 and ((self.root.entropy - curr_node.rchild.entropy) * 100 / self.root.entropy < self.reduction_threshold):
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

    def __build_sets(self):
        """
            This method is stubbed right now to return predetermined sets
            We hope to build a system that returns sets built by clustering words
            based on their semantic similarity and semantic relatedness in the given
            training corpus.

            todo : make no of sets and their size configurable if possible
        """

        try:
            data = json.loads(open(self.set_filename, "r").read())
        except IOError:
            logging.error("Set File not found")
            raise
        # Every element in the list needs to be a set because
        # belongs to operation utilises O(1) steps

        for i in range(len(data)):
            data[i] = set(data[i])
        return data

    def __serialize(self, filename):
        """
            Used for pickling and writing the constructed Trsl
            into a file for future use
        """

        open(filename, "wb").write(pickle.dumps(self.root))

    def __load(self, filename):
        """
            Used for loading the pickled data which
            is written in a file to the memory
        """

        self.root = pickle.loads(open(filename, "rb").read())

    def tree_walk(self, seed, no_of_words):
        """
            Used for random tree walk using the prediction
            from the constructed Trsl from a seed predictor variables
            upto no_of_words prediction
        """

        for index in range(no_of_words):
            dist = self.predict(seed[-(self.ngram_window_size-1)::])
            rand = random.random()
            sum = 0
            for i in dist.keys():
                sum += dist[i]
                if rand <= sum:
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
                        "LEVEL: %s, X%s = %s belongs to %s? YES",
                        steps, temp.predictor_variable_index,
                        predictor_variable_list[
                            temp.predictor_variable_index
                        ],
                        temp.set
                    )
                    temp = temp.lchild
                else:
                    logging.debug(
                        "LEVEL: %s, X%s = %s belongs to %s? NO",
                        steps, temp.predictor_variable_index,
                        predictor_variable_list[
                            temp.predictor_variable_index
                        ],
                        temp.set
                    )
                    temp = temp.rchild
            else:
                logging.info(
                    "Total Reduction in Entropy: %s -> %s%%",
                    self.root.entropy - temp.entropy,
                    100 * (self.root.entropy - temp.entropy)/self.root.entropy
                )
                logging.debug("Probable Distribution: %s", temp.dist)
                logging.info("Depth Reached: %s", steps)
                return temp.dist
