#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Tree Based Statistical language Model
    to be used in tandem with an acoustic model
    for speech recognition
"""


import ConfigParser
from collections import Counter, defaultdict
import json
import logging
import os
import random
import scipy.stats
import numpy as np
import trsl_preprocess
from node import Node
from pickling import PickleTrsl


class Trsl(object):

    """
        Trsl class implements a tree based statistical language model
        Arguments:
            self.reduction_threshold -> min reduction in entropy for a question
                                        at a node to further grow the tree
            self.ngram_window_size   -> no of predictor variables
                                        (inclusive of target)
    """

    def __init__(
            self,
            ngram_window_size=6,
            reduction_threshold=100,
            sample_size=10,
            model=None,
            corpus=None,
            config=None,
            set_filename=None,
            output_dir=None
            ):

        self.__init_logger()
        serialised_trsl = None
        if model is not None:
            # If model is provided for initialisation
            config = ConfigParser.RawConfigParser()
            if len(config.read(model)) > 0:
                set_filename = config.get('Trsl', 'set_filename')
                serialised_trsl = config.get('Trsl', 'serialised_trsl')
            else:
                # If model file is empty
                self.logger.error("Error Model file corrupted")
                return None
        else:
            # If model is not provided for initialisation
            if corpus is None:
                self.logger.error("""
                    Error, No pretrained model passed or corpus for
                    generating the model
                """)
                return None
            elif config is not None:
                # If config file is provided with corpus during initialisation
                config_parser = ConfigParser.RawConfigParser()
                if len(config_parser.read(config)) > 0:
                    # If set_filename is already passed, dont load from config
                    if set_filename is None:
                        set_filename = config_parser.get(
                            'Trsl', 'set_filename'
                        )
                        set_filename = None if (
                            set_filename == "None") else set_filename
                    ngram_window_size = config_parser.getint(
                        'Trsl', 'ngram_window_size'
                    )
                    sample_size = config_parser.getint(
                        'Trsl', 'sample_size'
                    )
                    reduction_threshold = config_parser.getfloat(
                        'Trsl', 'reduction_threshold'
                    )
                    self.no_of_clusters = config_parser.getint(
                        'Set', 'no_of_clusters'
                    )
                    self.no_of_words_set = config_parser.getint(
                        'Set', 'no_of_words'
                    )
                    self.word2vec_model_path = config_parser.get(
                        'Set', 'word2vec_model_path'
                    )
                else:
                    # If config file is empty
                    self.logger.error("Error config file corrupted")
                    return None
            elif config is None and set_filename is None:
                # If not set or config file passed for trsl initialisation
                self.logger.error("""
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
        self.sample_size = sample_size
        self.serialised_trsl = serialised_trsl
        self.word_ngram_table = None
        if output_dir is None and model is None:
            self.output_dir = "../data/models/" + self.filename.split("/")[-1]+"-model/"
        else:
            self.output_dir = output_dir
        self.__train()

    def __execute_scripts(self, script, error_msg):
        """
            The script which is passed is executed,
            as a seperate process, if execution failed,
            error message is displayed and exception raised.

        """

        status = os.system(script)
        if status is not 0:
            self.logger.error(error_msg)
            raise RuntimeError

    def __init_logger(self):
        """
            Initialise the logger for trsl
        """

        self.logger = logging.getLogger('Trsl')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        if self.logger.level is 0:
            self.logger.setLevel(logging.INFO)

    def __generate_sets(self):
        """
            Used to generate sets required for trsl building,
            preprocessing, vectorizing words, clustering vectors,
            using word2vec
        """

        if self.__is_set_building_required():
            selfvalues = dict(fname=self.filename, w2v_path=self.word2vec_model_path,
                              wordsets=self.no_of_words_set, clusters=self.no_of_clusters)

            scripts_folder = os.path.join(os.curdir, 'sets')
            preproc = os.path.join(scripts_folder, "preprocess_sets.py")
            word2vec = os.path.join(scripts_folder, "word_vectorizer.py")
            setbuild = os.path.join(scripts_folder, "set_building.py")

            selfvalues['preproc'] = preproc
            selfvalues['word2vec'] = word2vec
            selfvalues['setbuild'] = setbuild

            self.logger.info(
                "Preprocessing corpus for set building"
            )
            self.__execute_scripts(
                "python2 %(preproc)s %(fname)s" % selfvalues,
                "Preprocessing sets failed"
            )
            self.logger.info(
                "Generating word vectors from preprocessed data"
            )
            # self.__execute_scripts(
            #     "python2 %(word2vec)s %(fname)s-sorted %(fname)s-vectors %(w2v_path)s" % selfvalues,
            #     "Word vectors computing failed"
            # )

            file_path = self.output_dir
            # If folder does not exist, create the same
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            self.set_filename = os.path.join(file_path, 'sets')
            selfvalues['set_fname'] = self.set_filename

            self.logger.info("Performing Set Clustering")
            self.__execute_scripts(
                "python2 %(setbuild)s %(wordsets)s %(clusters)s %(fname)s-vectors %(set_fname)s" % selfvalues,
                "Set building operation failed"
            )

            # Remove intermediate temp files, 
            # frequency based sorted vocabulary and vectors for the vocabulary
            os.remove("%(fname)s-sorted" %selfvalues)
            os.remove("%(fname)s-vectors" %selfvalues)

    def __is_trsl_computed(self):
        """
            Returns True if precomputed trsl is utilised,
            when model is passed during initialisation
        """

        return False if self.serialised_trsl is None else True

    def __is_set_building_required(self):
        """
            Returns True if no model is passed or sets,
            thus sets needs to built at that scenario
        """

        return self.set_filename is None and self.config is not None

    def __train(self):
        """
            Given a filename(containing a training corpus),
            build a decision tree
            which can be accessed through self.root.
            if model is given, load the model directly
            if corpus is specified generate a model and store the same
        """

        # If model is passed with trsl, load the model directly
        if self.__is_trsl_computed():
            # check if the model file exists
            try:
                open(self.serialised_trsl, "r")
                self.logger.info("Loading precomputed trsl model")
                # Load precomputed trsl instance
                self.__load(self.serialised_trsl)
                self.logger.info(
                    "Reduction Threshold: %s",
                    self.reduction_threshold
                )
                self.logger.info("Corpus Filename: %s", self.filename)
                self.logger.info(
                    "Ngram Window Size: %s", self.ngram_window_size
                )
                self.logger.info("Set Utilised: %s", self.set_filename)
                self.logger.info("No of Samples: %s", self.sample_size)
            # if model file does not exists
            except (OSError, IOError) as e:
                self.logger.error("""
                    Serialised json file specified in the model missing,
                    precomputed trsl model could not be loaded :
                """ + str(e))
                return -1
        else:
            # If the model is not passed during initialisation
            self.__generate_sets()
            # Current initialisation of trsl is displayed
            self.logger.info(
                "Reduction Threshold: %s", self.reduction_threshold
            )
            self.logger.info("Corpus Filename: %s", self.filename)
            self.logger.info("Ngram Window Size: %s", self.ngram_window_size)
            self.logger.info("Set Utilised: %s", self.set_filename)
            self.logger.info("No of Samples: %s", self.sample_size)
            # If corpus is supplied and model needs to be generated
            self.word_sets = self.__build_sets()
            self.ngram_table, self.word_sets, self.word_ngram_table = (
                trsl_preprocess.preprocess(
                    self.filename, self.ngram_window_size, self.word_sets
                )
            )
            # Compute root node attributes
            self.__set_root_state()
            self.__process_node(self.root)
            self.current_leaf_nodes.append(self.root)
            stopping_criterion_msg = (
                "Tree growing stopped, reduction no longer improving accuracy"
            )
            # Process each node in trsl until a certain reduction threshold
            while not self.__stop_growing():
                # Pick the node with the highest reduction
                node_to_split = max(
                    self.current_leaf_nodes, key=lambda x: (
                        x.best_question.reduction
                    )
                )
                # If no reduction for the best question, stop tree growth
                if node_to_split.best_question.reduction == 0:
                    self.logger.debug(stopping_criterion_msg)
                    break
                # Split the node for the YES and NO path
                # process these child nodes individually
                self.__split_node(node_to_split)
                self.current_leaf_nodes.remove(node_to_split)
                self.__process_node(node_to_split.lchild)
                self.__process_node(node_to_split.rchild)
                self.current_leaf_nodes.append(node_to_split.lchild)
                self.current_leaf_nodes.append(node_to_split.rchild)

            self.logger.info("Total no of Nodes:" + str(self.no_of_nodes))

            # Compute word probability for all the leaf nodes
            self.__compute_word_probability()

            # Save the generated model with the serialised trsl

            file_path = self.output_dir

            if not os.path.exists(file_path):
                os.makedirs(file_path)
            self.__serialize(file_path + "serialised_trsl.json")
            config = ConfigParser.RawConfigParser()
            config.add_section('Trsl')
            config.set(
                'Trsl', 'serialised_trsl', file_path + 'serialised_trsl.json'
            )
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
        self.logger.debug(
            "Split Nodes at Level:%s",
            node_to_split.depth
        )

        # No further computations need to be done on this data
        if node_to_split is not self.root:
            node_to_split.parent.data_fragment = None
        # Best question has set index.
        # Get the set back and assign it to the node
        node_to_split.set = self.word_sets[node_to_split.best_question.set]
        node_to_split.predictor_variable_index = (
            node_to_split.best_question.predictor_variable_index
        )

        # YES path attributes for child node is set
        node_to_split.lchild = Node(self.ngram_window_size)
        node_to_split.lchild.set_known_predvars[
            node_to_split.predictor_variable_index
        ] = True
        node_to_split.lchild.parent = node_to_split
        node_to_split.lchild.data_fragment = (
            node_to_split.best_question.b_fragment
        )
        node_to_split.lchild.len_data_fragment = len(
            node_to_split.best_question.b_fragment
        )
        node_to_split.lchild.probability = (
            node_to_split.best_question.b_probability
        )
        node_to_split.lchild.absolute_entropy = (
            node_to_split.best_question.b_dist_entropy
        )
        node_to_split.lchild.probabilistic_entropy = (
            node_to_split.best_question.b_probability *
            node_to_split.best_question.b_dist_entropy
        )
        node_to_split.lchild.dist = node_to_split.best_question.b_dist
        node_to_split.lchild.depth = node_to_split.depth + 1

        # NO path attributes for child node is set
        node_to_split.rchild = Node(self.ngram_window_size)
        node_to_split.rchild.parent = node_to_split
        node_to_split.rchild.data_fragment = (
            node_to_split.best_question.nb_fragment
        )
        node_to_split.rchild.len_data_fragment = len(
            node_to_split.best_question.nb_fragment
        )
        node_to_split.rchild.probability = (
            node_to_split.best_question.nb_probability
        )
        node_to_split.rchild.absolute_entropy = (
            node_to_split.best_question.nb_dist_entropy
        )
        node_to_split.rchild.probabilistic_entropy = (
            node_to_split.best_question.nb_probability *
            node_to_split.best_question.nb_dist_entropy
        )
        node_to_split.rchild.dist = node_to_split.best_question.nb_dist
        node_to_split.rchild.depth = node_to_split.depth + 1

    def __stop_growing(self):
        """
            Stop the tree growth if the reduction threshold
            is reached from the root node to the leaf nodes
            Return Type:
                True    -> Stop tree growth
                False   -> Continue tree growth
        """

        probabilistic_entropies_sum = sum(
            node.probabilistic_entropy for node in self.current_leaf_nodes
        )
        entropy_reduction = (
            self.root.absolute_entropy - probabilistic_entropies_sum
        )
        entropy_reduction_percentage = (
            entropy_reduction * 100 / self.root.absolute_entropy
        )
        if entropy_reduction_percentage < self.reduction_threshold:
            reduction_from_root = (
                self.root.absolute_entropy - probabilistic_entropies_sum
            )
            self.logger.debug(
                "Reduction from Root: %s %%",
                reduction_from_root * 100 / self.root.absolute_entropy
            )
            return False
        else:
            self.logger.debug(
                "Reduction Threshold reached, stopping tree growth!"
            )
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
        self.root.data_fragment = np.array(
            self.root.data_fragment, dtype='int32'
        )
        # Compute root node entropy, probability
        target_word_column = self.root.data_fragment[
            :, (self.ngram_window_size - 1)
        ]
        probabilities = (
            np.bincount(
                target_word_column
            ).astype('float32') / target_word_column.shape[0]
        )
        self.root.absolute_entropy = scipy.stats.entropy(probabilities, base=2)
        self.root.probabilistic_entropy = (
            self.root.absolute_entropy * self.root.probability
        )
        self.root.len_data_fragment = len(self.root.data_fragment)
        self.logger.debug(
            "Root Entropy: %s",
            self.root.probabilistic_entropy
        )

    def __generate_pred_var_set_pairs(self):
        """
            Lazy generator for Predictor, Set combinations
        """

        for x_index in xrange(0, self.ngram_window_size - 1):
            for set_index in xrange(len(self.word_sets)):
                yield (x_index, set_index)

    def __process_node(self, curr_node):
        """
            Used to process curr_node by computing best reduction
            and choosing to create children nodes or not based on
            self.reduction_threshold
        """

        # Pick the node with min avg_conditional_entropy
        curr_node.best_question = min(
            (question for question in curr_node.generate_questions(
                self.ngram_table,
                self.__generate_pred_var_set_pairs
            )),
            key=lambda question: question.avg_conditional_entropy if (
                len(question.b_fragment) > self.sample_size) and (
                    len(question.nb_fragment) > self.sample_size
            ) else float('inf')
        )

        if ((len(curr_node.best_question.b_fragment) <= self.sample_size) or
                (len(curr_node.best_question.b_fragment) <= self.sample_size)):
            curr_node.best_question.reduction = 0

        self.logger.debug(
            "Reduction: %s, (%s,%s)",
            curr_node.best_question.reduction,
            len(curr_node.best_question.b_fragment),
            len(curr_node.best_question.nb_fragment)
        )

    def __build_sets(self):
        """
            We hope to build a system that returns sets
            built by clustering words based on their semantic similarity
            and semantic relatedness in the given
            training corpus.

            todo : make no of sets and their size configurable if possible
        """

        try:
            with open(self.set_filename, "r") as sets_file:
                data = json.loads(sets_file.read())
        except IOError as e:
            self.logger.error("Set File not found :" + str(e))
            raise IOError
        # Every element in the list needs to be a set because
        # belongs to operation utilises O(1) steps

        return map(lambda x: set(x), data)

    def __serialize(self, filename):
        """
            Used for pickling and writing the constructed Trsl
            into a file for future use
        """

        with open(filename, "w") as serialised_trsl_file:
            serialised_trsl_file.write(PickleTrsl().serialise(self))

    def __load(self, filename):
        """
            Used for loading the pickled data which
            is written in a file to the memory
        """

        with open(filename, "r") as serialised_trsl_file:
            data = serialised_trsl_file.read()
            PickleTrsl().deserialise(self, data)

    def tree_walk(self, seed, no_of_words):
        """
            Used for random tree walk using the prediction
            from the constructed Trsl from a seed predictor variables
            upto no_of_words prediction
        """

        for _ in xrange(no_of_words):
            dist = self.predict(seed[-(self.ngram_window_size - 1)::])
            rand = random.random()
            s = 0
            for i in dist.keys():
                s += dist[i]
                if rand <= s:
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
            if node.word_probability is not None:
                sum_frequences = sum(node.word_probability.values())
                node.word_probability = {
                    tup[0]: tup[1] / float(
                        sum_frequences
                    ) for tup in node.word_probability.items()
                }

    def predict(self, predictor_variable_list):
        """
            Given a list of predictor words, this method
            returns the probability distribution of the
            next words that could occur next.

            todo: exception handling when predict is called before train
        """

        if len(predictor_variable_list) != self.ngram_window_size - 1:
            self.logger.error("""
                predictor_variable_list size should conform
                with ngram window size
                """)
            return None
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
                # Since the decision tree is a full binary tree,
                # both children exist, single check suffices.
                predictor_variable = (
                    predictor_variable_list[temp.predictor_variable_index]
                )
                if predictor_variable in temp.set:
                    temp = temp.lchild
                # if node is leaf node
                else:
                    temp = temp.rchild
            else:
                return temp
