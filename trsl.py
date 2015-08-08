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
import random
import pickle
from pickling import PickleTrsl
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
            set_filename=None,
            samples = None
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
                        set_filename = config.get('Trsl', 'set_filename')
                    if samples is None:
                        samples = int(config.get('Trsl', 'samples'))
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
        logging.info("No of Samples: %s", samples)
        self.reduction_threshold = reduction_threshold
        self.ngram_window_size = ngram_window_size
        self.root = Node(ngram_window_size)
        self.previous_best_questions = set()
        self.no_of_nodes = 1
        self.ngram_table = None
        self.vocabulary_set = None
        self.word_sets = None
        self.current_leaf_nodes = []
        self.max_depth = 0
        self.min_depth = float('inf')
        self.filename = filename
        self.set_filename = set_filename
        self.samples = samples

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
            self.word_sets = self.__build_sets()
            self.ngram_table, self.word_ngram_table, self.word_sets, self.set_reverse_index = preprocess.preprocess(
                self.filename, self.ngram_window_size, self.word_sets
            )
            self.__set_root_state()
            self.__process_node(self.root)
            self.current_leaf_nodes.append(self.root)
            while not self.__stop_growing():
                node_to_split = max(self.current_leaf_nodes, key=lambda x: x.best_question.reduction)
                if node_to_split.best_question.reduction == 0:
                    logging.debug("Tree growing stopped, reduction no longer improving accuracy")
                    break
                self.__split_node(node_to_split)
                self.current_leaf_nodes.remove(node_to_split)
                self.__process_node(node_to_split.lchild)
                self.__process_node(node_to_split.rchild)
                self.current_leaf_nodes.append(node_to_split.lchild)
                self.current_leaf_nodes.append(node_to_split.rchild)

            for leaf in self.current_leaf_nodes:
                leaf.dist = self.__calculate_word_dist(leaf)

            logging.info("Total no of Nodes:"+ str(self.no_of_nodes))
            logging.info(
                "Max Depth: %s Min Depth: %s",
                self.max_depth,
                self.min_depth
            )
            self.__serialize(self.filename + ".dat")

    def __calculate_word_dist(self, leaf):

        dist = {}
        for i in leaf.row_fragment_indices:
             target_word = self.word_ngram_table[
                 i, self.word_ngram_table.ngram_window_size-1
             ]
             try:
                 dist[target_word] += 1.0
             except KeyError:
                 dist[target_word] = 1.0

        frequency_sum = sum(dist.values())

        for key in dist.keys():
            dist[key] /= frequency_sum

        return dist

    def __split_node(self, node_to_split):

        self.no_of_nodes += 2
        logging.debug(
            "Split Nodes at Level:%s"
            % (
                node_to_split.depth
            )
        )
        #Best question has set index. Get the set back and assign it to the node
        node_to_split.set = self.word_sets[node_to_split.best_question.set]
        node_to_split.predictor_variable_index = (
            node_to_split.best_question.predictor_variable_index
        )
        node_to_split.lchild = Node(self.ngram_window_size)
        node_to_split.lchild.set_known_predvars[node_to_split.predictor_variable_index] = True
        node_to_split.lchild.parent = node_to_split
        node_to_split.lchild.row_fragment_indices = node_to_split.best_question.b_indices
        node_to_split.lchild.probability = node_to_split.best_question.b_probability
        node_to_split.lchild.absolute_entropy = node_to_split.best_question.b_dist_entropy
        node_to_split.lchild.probabilistic_entropy = node_to_split.best_question.b_probability * node_to_split.best_question.b_dist_entropy
        node_to_split.lchild.dist = node_to_split.best_question.b_dist
        node_to_split.lchild.depth = node_to_split.depth + 1

        node_to_split.rchild = Node(self.ngram_window_size)
        node_to_split.rchild.parent = node_to_split
        node_to_split.rchild.row_fragment_indices = node_to_split.best_question.nb_indices
        node_to_split.rchild.probability = node_to_split.best_question.nb_probability
        node_to_split.rchild.absolute_entropy = node_to_split.best_question.nb_dist_entropy
        node_to_split.rchild.probabilistic_entropy = node_to_split.best_question.nb_probability * node_to_split.best_question.nb_dist_entropy
        node_to_split.rchild.dist = node_to_split.best_question.nb_dist
        node_to_split.rchild.depth = node_to_split.depth + 1

    def __stop_growing(self):

        probabilistic_entropies_sum = sum(node.probabilistic_entropy for node in self.current_leaf_nodes)
        if (self.root.absolute_entropy - probabilistic_entropies_sum) * 100 / self.root.absolute_entropy  < self.reduction_threshold:
            logging.debug(
                "Reduction from Root: %s %%"
                % (
                    (self.root.absolute_entropy - probabilistic_entropies_sum) * 100 / self.root.absolute_entropy
                )
            )
            return False
        else:
            logging.debug("Reduction Threshold reached, stopping tree growth!")
            return True

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
        self.root.probability = 1
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
            self.root.absolute_entropy += -probability_of_info_gain

        self.root.probabilistic_entropy = self.root.absolute_entropy * self.root.probability
        logging.debug(
            "Root Entropy: %s",
            self.root.probabilistic_entropy
        )

    def __generate_pred_var_set_pairs(self):
        """
            Lazy generator for Predictor, Set combinations
        """

        for x_index in range(0, self.ngram_window_size-1):
            for set_index in range(len(self.word_sets)):
                yield (x_index, set_index)

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


        #bind ngramtable to a partial function
        eval_question = partial(curr_node.eval_question, self.ngram_table)
        questions = map(eval_question, self.__generate_pred_var_set_pairs())
        curr_node.best_question = min(questions, key=lambda question: question.avg_conditional_entropy if len(question.b_indices) > self.samples and len(question.nb_indices) > self.samples else float('inf'))
        if len(curr_node.best_question.b_indices) <= self.samples or len(curr_node.best_question.nb_indices) <= self.samples:
            curr_node.reduction = 0
        else:
            logging.debug("Reduction: %s, (%s,%s)"
                %(
                    curr_node.best_question.reduction,
                    len(curr_node.best_question.b_indices),
                    len(curr_node.best_question.nb_indices)
                )
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

        logging.info("Serialised Data and stored as " + self.filename+".dat")
        open(filename, "w").write(PickleTrsl().serialise(self))
        #open(filename, "wb").write(pickle.dumps(self.root))

    def __load(self, filename):
        """
            Used for loading the pickled data which
            is written in a file to the memory
        """

        f = open(self.filename+".dat","r")
        data = f.read()
        f.close()
        PickleTrsl().deserialise(self, data)
        #self.root = pickle.loads(open(filename, "rb").read())

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
                    self.root.probabilistic_entropy - temp.probabilistic_entropy,
                    100 * (self.root.probabilistic_entropy - temp.probabilistic_entropy)/self.root.probabilistic_entropy
                )
                logging.debug("Probable Distribution: %s", temp.dist)
                logging.info("Depth Reached: %s", steps)
                return temp.dist
