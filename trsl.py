#! /usr/bin/env/python2
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Tree Based Statistical language Model
    to be used in tandem with an acoustic model
    for speech recognition
"""


from collections import Counter, defaultdict
from node import Node
import json
import logging
import preprocess   # todo check for namespace pollution
import random
import os
from pickling import PickleTrsl
import ConfigParser
import scipy.stats
import numpy as np

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
            ngram_window_size=6,
            reduction_threshold=100,
            samples=10,
            model=None,
            corpus=None,
            config=None,
            set_filename=None
        ):

        serialised_trsl = None
        if model is not None:
            # If model is provided for initialisation
            config = ConfigParser.RawConfigParser()
            if len(config.read(model)) > 0:
                set_filename = config.get('Trsl', 'set_filename')
                serialised_trsl = config.get('Trsl', 'serialised_trsl')
            else:
                # If model file is empty
                logging.error("Error Model file corrupted")
                return None
        else:
            # If mode is not provided for initialisation
            if corpus is None:
                logging.error("""
                    Error, No pretrained model passed or corpus for
                    generating the model
                """)
                return None
            elif config is not None:
                # If config file is provided with the corpus during initialisation
                config_parser = ConfigParser.RawConfigParser()
                if len(config_parser.read(config)) > 0:
                    # If set_filename is already passed, dont load from config
                    if set_filename is None:
                        set_filename = config_parser.get('Trsl', 'set_filename')
                        set_filename = None if set_filename == "None" else set_filename
                    ngram_window_size = config_parser.getint('Trsl', 'ngram_window_size')
                    samples = config_parser.getint('Trsl', 'samples')
                    reduction_threshold = config_parser.getfloat('Trsl', 'reduction_threshold')
                    self.no_of_clusters = config_parser.getint('Set', 'no_of_clusters')
                    self.no_of_words_set = config_parser.getint('Set', 'no_of_words')
                    self.word2vec_model_path = config_parser.get('Set', 'word2vec_model_path')
                else:
                    # If config file is empty
                    logging.error("Error config file corrupted")
                    return None
            elif config is None and set_filename is None:
                # If not set or config file passed for trsl initialisation
                logging.error("""
                    Error, No precomputed sets passed, or config file
                    to generate sets passed
                """)
                return None

        self.config = config
        self.reduction_threshold = reduction_threshold
        self.ngram_window_size = ngram_window_size
        self.root = Node(ngram_window_size)
        self.previous_best_questions = set()
        self.no_of_nodes = 1
        self.ngram_table = None
        self.vocabulary_set = None
        self.word_sets = None
        self.current_leaf_nodes = []
        self.filename = corpus
        self.set_filename = set_filename
        self.samples = samples
        self.serialised_trsl = serialised_trsl
        self.word_ngram_table = None

    def train(self):
        """
            Given a filename(containing a training corpus), build a decision tree
            which can be accessed through self.root.
            if model is given, load the model directly
            if corpus is specified generate a model and store the same
        """

        # If model is passed with trsl, load the model directly
        if self.serialised_trsl is not None:
            # check if the model file exists
            try:
                open(self.serialised_trsl, "r")
                logging.info("Loading precomputed trsl model")
                # Load precomputed trsl instance
                self.__load(self.serialised_trsl)
                logging.info("Reduction Threshold: %s", self.reduction_threshold)
                logging.info("Corpus Filename: %s", self.filename)
                logging.info("Ngram Window Size: %s", self.ngram_window_size)
                logging.info("Set Utilised: %s", self.set_filename)
                logging.info("No of Samples: %s", self.samples)
            # if model file does not exists
            except (OSError, IOError):
                logging.error("""
                    Serialised json file specified in the model missing,
                    precomputed trsl model could not be loaded.
                """)
                return -1
        else:
            # If the model is not passed during initialisation
            if self.set_filename is None and self.config is not None:
                # If no set is passed, but config is passed to build sets
                logging.info("Building sets from the Corpus")
                status = os.system(
                    "python2 ./sets/preprocess_sets.py " + self.filename
                )
                if status is 0:
                    # Preprocess of corpus for sets successfull
                    logging.info("Corpus preprocessed for set building")
                    status = os.system(
                        "python2 ./sets/word-vectorizer.py "
                        + self.filename + "-sorted "
                        + self.filename + "-vectors "
                        + self.word2vec_model_path
                    )
                else:
                    # Preprocess of corpus failed
                    logging.error("Preprocessing sets failed")
                    return None
                if status is 0:
                    # Computing word vectors for the preprocessed data successfull
                    logging.info("Word vectors computed")
                    file_path = "./"+self.filename.split("/")[-1]+"-model/"
                    # If folder does not exist, create the same
                    if not os.path.exists(file_path):
                        os.makedirs(file_path)
                    set_filename = file_path + "sets"
                    status = os.system(
                        "python2 ./sets/set_building.py "
                        + str(self.no_of_words_set) + " "
                        + str(self.no_of_clusters) + " "
                        + self.filename+"-vectors " + set_filename
                    )
                    if status is 0:
                        # Computing clusters from the vectors produced
                        logging.info("Set clustering completed")
                        self.set_filename = set_filename
                else:
                    # Computing word vectors for the preprocessed data failed
                    logging.error("Word Vectorizing failed")
                    return None
                if status is not 0:
                    # Computing clusters from vectors failed
                    logging.error("Set Clustering Failed")
                    return None
            # Current initialisation of trsl is displayed
            logging.info("Reduction Threshold: %s", self.reduction_threshold)
            logging.info("Corpus Filename: %s", self.filename)
            logging.info("Ngram Window Size: %s", self.ngram_window_size)
            logging.info("Set Utilised: %s", self.set_filename)
            logging.info("No of Samples: %s", self.samples)
            # If corpus is supplied and model needs to be generated
            self.word_sets = self.__build_sets()
            self.ngram_table, self.word_sets, self.word_ngram_table = preprocess.preprocess(
                self.filename, self.ngram_window_size, self.word_sets
            )
            # Compute root node attributes
            self.__set_root_state()
            self.__process_node(self.root)
            self.current_leaf_nodes.append(self.root)
            # Process each node in trsl until a certain reduction threshold
            while not self.__stop_growing():
                # Pick the node with the highest reduction
                node_to_split = max(
                    self.current_leaf_nodes, key=lambda x: x.best_question.reduction
                )
                # If no reduction for the best question, stop tree growth
                if node_to_split.best_question.reduction == 0:
                    logging.debug("""Tree growing stopped,
                        reduction no longer improving accuracy
                    """)
                    break
                # Split the node for the YES and NO path, process them individually
                self.__split_node(node_to_split)
                self.current_leaf_nodes.remove(node_to_split)
                self.__process_node(node_to_split.lchild)
                self.__process_node(node_to_split.rchild)
                self.current_leaf_nodes.append(node_to_split.lchild)
                self.current_leaf_nodes.append(node_to_split.rchild)

            logging.info("Total no of Nodes:"+ str(self.no_of_nodes))

            # Compute word probability for all the leaf nodes
            self.__compute_word_probability()

            # Save the generated model with the serialised trsl
            file_path = "./"+self.filename.split("/")[-1]+"-model/"
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            self.__serialize(file_path + "serialised_trsl.json")
            config = ConfigParser.RawConfigParser()
            config.add_section('Trsl')
            config.set('Trsl', 'serialised_trsl', file_path + 'serialised_trsl.json')
            config.set('Trsl', 'set_filename', self.set_filename)
            with open(file_path + 'model', 'w') as model:
                config.write(model)

    def __split_node(self, node_to_split):
        """
            Receives the node to split, and creates two nodes as children
            with the specific attributes from the node based on YES and
            NO path
        """

        self.no_of_nodes += 2
        logging.debug(
            "Split Nodes at Level:%s"
            % (
                node_to_split.depth
            )
        )

        #No further computations need to be done on this data
        if node_to_split is not self.root:
            node_to_split.parent.data_fragment = None
        #Best question has set index. Get the set back and assign it to the node
        node_to_split.set = self.word_sets[node_to_split.best_question.set]
        node_to_split.predictor_variable_index = (
            node_to_split.best_question.predictor_variable_index
        )

        # YES path attributes for child node is set
        node_to_split.lchild = Node(self.ngram_window_size)
        node_to_split.lchild.set_known_predvars[node_to_split.predictor_variable_index] = True
        node_to_split.lchild.parent = node_to_split
        node_to_split.lchild.data_fragment = node_to_split.best_question.b_fragment
        node_to_split.lchild.len_data_fragment = len(node_to_split.best_question.b_fragment)
        node_to_split.lchild.probability = node_to_split.best_question.b_probability
        node_to_split.lchild.absolute_entropy = node_to_split.best_question.b_dist_entropy
        node_to_split.lchild.probabilistic_entropy = (
            node_to_split.best_question.b_probability *
            node_to_split.best_question.b_dist_entropy
        )
        node_to_split.lchild.dist = node_to_split.best_question.b_dist
        node_to_split.lchild.depth = node_to_split.depth + 1

        # NO path attributes for child node is set
        node_to_split.rchild = Node(self.ngram_window_size)
        node_to_split.rchild.parent = node_to_split
        node_to_split.rchild.data_fragment = node_to_split.best_question.nb_fragment
        node_to_split.rchild.len_data_fragment = len(node_to_split.best_question.nb_fragment)
        node_to_split.rchild.probability = node_to_split.best_question.nb_probability
        node_to_split.rchild.absolute_entropy = node_to_split.best_question.nb_dist_entropy
        node_to_split.rchild.probabilistic_entropy = (
            node_to_split.best_question.nb_probability *
            node_to_split.best_question.nb_dist_entropy
        )
        node_to_split.rchild.dist = node_to_split.best_question.nb_dist
        node_to_split.rchild.depth = node_to_split.depth + 1

    def __stop_growing(self):
        """
            Stop the tree growth if the reduction threshold is reached
            from the root node to the leaf nodes
            Return Type:
                True    -> Stop tree growth
                False   -> Continue tree growth
        """

        probabilistic_entropies_sum = sum(
            node.probabilistic_entropy for node in self.current_leaf_nodes
        )
        entropy_reduction = self.root.absolute_entropy - probabilistic_entropies_sum
        entropy_reduction_percentage = entropy_reduction * 100 / self.root.absolute_entropy
        if entropy_reduction_percentage < self.reduction_threshold:
            logging.debug(
                "Reduction from Root: %s %%"
                % (
                    (
                        self.root.absolute_entropy - probabilistic_entropies_sum
                    ) * 100 / self.root.absolute_entropy
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
            Also, adds all ngram windows into data_fragment,
            since all the data is present at the root of a decision tree
        """

        self.root.probability = 1
        self.root.data_fragment = []
        for sequence in self.ngram_table.generate_all_ngrams():
            self.root.data_fragment.append(sequence)
        self.root.data_fragment = np.array(self.root.data_fragment, dtype='int32')
        # Compute root node entropy, probability
        target_word_column = self.root.data_fragment[:, (self.ngram_window_size - 1)]
        probabilities = (
            np.bincount(target_word_column).astype('float32') / target_word_column.shape[0]
        )
        self.root.absolute_entropy = scipy.stats.entropy(probabilities, base=2)
        self.root.probabilistic_entropy = self.root.absolute_entropy * self.root.probability
        self.root.len_data_fragment = len(self.root.data_fragment)
        logging.debug(
            "Root Entropy: %s",
            self.root.probabilistic_entropy
        )

    def __generate_pred_var_set_pairs(self):
        """
            Lazy generator for Predictor, Set combinations
        """

        for x_index in range(0, self.ngram_window_size-1):
            for set_index in xrange(len(self.word_sets)):
                yield (x_index, set_index)

    def __process_node(self, curr_node):
        """
            Used to process curr_node by computing best reduction
            and choosing to create children nodes or not based on
            self.reduction_threshold
        """


        # Pick the node which gives you the least avg_conditional_entropy in the child nodes
        curr_node.best_question = min(
            (question for question in curr_node.generate_questions(
                self.ngram_table,
                self.__generate_pred_var_set_pairs
            )),
            key=lambda question: question.avg_conditional_entropy if len(question.b_fragment) > self.samples and len(question.nb_fragment) > self.samples else float('inf')
        )

        if ((len(curr_node.best_question.b_fragment) <= self.samples) or
                (len(curr_node.best_question.b_fragment) <= self.samples)):
            curr_node.reduction = 0
        else:
            logging.debug(
                "Reduction: %s, (%s,%s)"
                %(
                    curr_node.best_question.reduction,
                    len(curr_node.best_question.b_fragment),
                    len(curr_node.best_question.nb_fragment)
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

        for i in xrange(len(data)):
            data[i] = set(data[i])
        return data

    def __serialize(self, filename):
        """
            Used for pickling and writing the constructed Trsl
            into a file for future use
        """

        open(filename, "w").write(PickleTrsl().serialise(self))

    def __load(self, filename):
        """
            Used for loading the pickled data which
            is written in a file to the memory
        """

        serialised_trsl = open(filename, "r")
        data = serialised_trsl.read()
        serialised_trsl.close()
        PickleTrsl().deserialise(self, data)

    def tree_walk(self, seed, no_of_words):
        """
            Used for random tree walk using the prediction
            from the constructed Trsl from a seed predictor variables
            upto no_of_words prediction
        """

        for index in xrange(no_of_words):
            dist = self.predict(seed[-(self.ngram_window_size-1)::])
            rand = random.random()
            sum = 0
            for i in dist.keys():
                sum += dist[i]
                if rand <= sum:
                    seed.append(i)
                    break

        return seed


    def __compute_word_probability(self):
        """
            Compute word probability for all the leaf nodes,
            generate ngram sequences from the corpus, compute target
            word frequency from which probability is computed for each
            leaf node
        """

        for ngram_sequence in self.word_ngram_table.generate_all_ngrams():
            target_word = ngram_sequence[-1]
            node = self.__traverse_trsl(ngram_sequence[:-1])
            if node.word_probability is None:
                node.word_probability = defaultdict(lambda: 0)
            node.word_probability[target_word] += 1
        for node in self.current_leaf_nodes:
            if node.word_probability is None:
                continue
            else:
                sum_frequences = sum(node.word_probability.values())
                node.word_probability = {
                    tup[0]:tup[1]/float(sum_frequences) for tup in node.word_probability.iteritems()
                }


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
        else:
            temp = self.__traverse_trsl(predictor_variable_list)
            return Counter(temp.word_probability)

    def __traverse_trsl(self, predictor_variable_list):
        """
            Traverse the trsl based on the input
            predictor_variable_list, return the leaf node
            which is reached.
            Return type
                Node [leaf node]
        """

        temp = self.root
        while True:
            # if node is internal node
            if temp.rchild is not None:
                # Since the decision tree is  a full binary tree, both children exist
                if predictor_variable_list[temp.predictor_variable_index] in temp.set:
                    temp = temp.lchild
                # if node is leaf node
                else:
                    temp = temp.rchild
            else:
                return temp
