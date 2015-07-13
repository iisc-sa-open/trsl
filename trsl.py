#! /usr/bin/env/python2
"""
"""


from collections import Counter   # todo : remove this
from node import Node
import preprocess   # todo check for namespace pollution
import Queue
import math
import logging


class Trsl():
    """
    """

    def __init__(self, ngram_window_size=5, reduction_threshold=0.15):

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

    def train(self, filename):
        """
            todo : make sure the question is unique
            from the parent all the way upto the root
        """

        self.ngram_table, self.vocabulary_set = preprocess.preprocess(
            filename, self.ngram_window_size
        )
        self.set_root_state()
        self.node_queue.put(self.root)
        self.word_sets = self.build_sets()
        while(not self.node_queue.empty()):
            self.process_node(self.node_queue.get())
        print("No of Nodes:", self.no_of_nodes)

    def set_root_state(self):

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

    def process_node(self, current_node):
        """
            Used to process current_node by computing best reduction
            and choosing to create children nodes or not based on
            reduction_threshold

            Naming Convention:
                *    nb  -> Not belongs the the selected set
                *    b   -> Belongs to the selected set
                *    Xi  -> Predictive variable index
                *   Si  -> Set index
                *   cnb -> current node not belongs to the selected set
                *   cb  -> current node belongs to the set
        """

        global logging

        best_question_data = {
            'b_indices': [],
            'nb_indices': [],
            'b_dist': {},
            'nb_dist': {},
            'b_dist_entropy': 0,
            'nb_dist_entropy': 0,
            'reduction': 0,
            'set': set(),
            'predictor_variable_index': 0
        }
        self.no_of_nodes += 1
        for Xi in range(0, self.ngram_window_size-1):
            for Si in self.word_sets:
                if self.question_already_asked(current_node, Xi, Si):
                    continue
                cb_dist = {}
                cnb_dist = {}
                current_reduction = 0
                cb_indices = []
                cnb_indices = []
                cb_dist_entropy = 0
                cnb_dist_entropy = 0
                for table_index in current_node.row_fragment_indices:
                    predictor_word = self.ngram_table[table_index, Xi]
                    target_word = self.ngram_table[
                        table_index, self.ngram_window_size-1
                    ]
                    if predictor_word in Si:
                        cb_indices.append(table_index)
                        try:
                            cb_dist[target_word] += 1.0
                        except KeyError:
                            cb_dist[target_word] = 1.0
                    else:
                        cnb_indices.append(table_index)
                        try:
                            cnb_dist[target_word] += 1.0
                        except KeyError:
                            cnb_dist[target_word] = 1.0
                b_frequency_sum = sum(cb_dist.values())
                for key in cb_dist.keys():
                    frequency = cb_dist[key]
                    probability = frequency/b_frequency_sum
                    probability_of_info_gain = (
                        probability * math.log(probability, 2)
                    )
                    cb_dist[key] = probability
                    cb_dist_entropy += -probability_of_info_gain

                nb_frequency_sum = sum(cnb_dist.values())
                for key in cnb_dist.keys():
                    frequency = cnb_dist[key]
                    probability = frequency/nb_frequency_sum
                    probability_of_info_gain = (
                        probability * math.log(probability, 2)
                    )
                    cnb_dist[key] = probability
                    cnb_dist_entropy += -probability_of_info_gain
                size_row_fragment_indices = (
                    len(current_node.row_fragment_indices)
                )
                b_probability = (
                    float(len(cb_indices))/size_row_fragment_indices
                )
                nb_probability = (
                    float(len(cnb_indices))/size_row_fragment_indices
                )
                current_average_conditional_entropy = (
                    b_probability * cb_dist_entropy + nb_probability
                    * cnb_dist_entropy
                )
                current_reduction = (
                    current_node.entropy - current_average_conditional_entropy
                )
                #print(Xi,Si,current_reduction)
                if best_question_data['reduction'] < current_reduction:
                    best_question_data['reduction'] = current_reduction
                    best_question_data['b_indices'] = cb_indices
                    best_question_data['b_dist'] = cb_dist
                    best_question_data['b_dist_entropy'] = (
                        cb_dist_entropy
                    )
                    best_question_data['nb_indices'] = cnb_indices
                    best_question_data['nb_dist'] = cnb_dist
                    best_question_data['nb_dist_entropy'] = (
                        cnb_dist_entropy
                    )
                    best_question_data['set'] = Si
                    best_question_data['predictor_variable_index'] = Xi

        if best_question_data['reduction'] > self.reduction_threshold:
            current_node.set = best_question_data['set']
            current_node.predictor_variable_index = best_question_data['predictor_variable_index']
            current_node.lchild = Node()
            current_node.lchild.row_fragment_indices = best_question_data['b_indices']
            current_node.lchild.entropy = best_question_data['b_dist_entropy']
            current_node.lchild.dist = best_question_data['b_dist']
            current_node.rchild = Node()
            current_node.rchild.row_fragment_indices = best_question_data['nb_indices']
            current_node.rchild.entropy = best_question_data['nb_dist_entropy']
            current_node.rchild.dist = best_question_data['nb_dist']
            self.node_queue.put(current_node.lchild)
            self.node_queue.put(current_node.rchild)
            current_node.lchild.parent = current_node
            current_node.rchild.parent = current_node

        else:
            logging.info(
                "Leaf Node reached, Top Probability dist:%s",
                dict(Counter(current_node.dist).most_common(5))
            )

    def question_already_asked(self, current_node, Xi, Si):

        parent = current_node.parent
        while parent is not None:
            if parent.set == Si and parent.predictor_variable_index == Xi:
                return True
            else:
                parent = parent.parent
        return False

    def build_sets(self):

        """
            calculate relative distance between words
            cluster words into n categories
            todo : make no of sets configurable if possible
            todo :
        """

        return [
            set(["the","for","in","at","a"])
            #set(["that", "it", "as", "he", "for"]),
            #set(["am", "only", "if", "little", "when"]),
            #set(["and", "of", "a", "was", "in"]),
            #set(["the", "to", "and", "of", "a"])
        ]

    def predict(self, predictor_variable_list):

        if len(predictor_variable_list) != self.ngram_window_size-1:
            raise ValueError(
                """predictor_variable_list size
                should conform with ngram window size"""
            )
        temp = self.root
        steps = 0
        while(True):
            if temp.rchild is not None:
                steps += 1
                # since its a binary tree, both children exist
                if predictor_variable_list[temp.predictor_variable_index] in temp.set:
                    print(
                        "LEVEL: %s, X%s = %s belongs to %s? YES"
                        % (
                            steps, temp.predictor_variable_index,
                            predictor_variable_list[temp.predictor_variable_index],
                            temp.set
                        )
                    )
                    temp = temp.lchild
                else:
                    print(
                        "LEVEL: %s, X%s = %s belongs to %s? NO"
                        % (
                            steps, temp.predictor_variable_index,
                            predictor_variable_list[temp.predictor_variable_index],
                            temp.set
                        )
                    )
                    temp = temp.rchild
            else:
                return temp.dist


def init_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    logging.basicConfig(filename='trsl.log', filemode='w', level=logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
