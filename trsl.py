#! /usr/bin/env/python2

from collections import Counter # todo : remove this
from node import Node
import preprocess # todo check for namespace pollution
import time
import Queue
import math
import logging

class Trsl():

	def __init__(self, ngram_window_size=5, reduction_threshold=0.5):

		self.reduction_threshold = reduction_threshold
		self.ngram_window_size = ngram_window_size
		self.root = Node()
		self.previous_best_questions = set()
		self.node_queue = Queue.Queue(-1) # inf queue size hopefully <- todo : evaluate if we need this

	def train(self, filename):
		"""
		todo : make sure the question is unique from the parent all the way upto the root 
		"""
		self.ngram_table, self.vocabulary_set = preprocess.preprocess(filename, self.ngram_window_size)
		self.set_root_state()
		self.node_queue.put(self.root)
		self.word_sets = self.build_sets()
		while(not self.node_queue.empty()):
			self.process_node(self.node_queue.get())

	def set_root_state(self):

		self.root.distribution = {}
		for index in range(0,len(self.ngram_table)):
			try:
				self.root.distribution[self.ngram_table[index][self.ngram_window_size-1]] += 1.0
			except KeyError:
				self.root.distribution[self.ngram_table[index][self.ngram_window_size-1]] = 1.0

		self.root.ngram_fragment_row_indices = [x for x in range(0, len(self.ngram_table))] # todo check if computes through entrie ds or not

		for key in self.root.distribution.keys():
			frequency = self.root.distribution[key]
			probability = frequency/len(self.root.ngram_fragment_row_indices)
			probability_of_info_gain = probability * math.log(probability,2)
			self.root.distribution[key] = probability
			self.root.entropy += -probability_of_info_gain

		
	def process_node(self, current_node):
		"""
		"""
		global logging

		best_question_data = {
			'belongs_to_indices': [],
			'not_belongs_to_indices': [],
			'belongs_to_distribution': {},
			'not_belongs_to_distribution': {},
			'belongs_to_distribution_entropy': 0,
			'not_belongs_to_distribution_entropy': 0,
			'reduction': 0,
			'set': set(),
			'predictor_variable_index': 0
		}

		for Xi in range(0, self.ngram_window_size-2):
			for Si in self.word_sets:
				if self.question_already_asked(current_node, Xi, Si):
					continue
				current_belongs_to_distribution = {}
				current_not_belongs_to_distribution = {}
				current_reduction = 0
				current_belongs_to_indices = []
				current_not_belongs_to_indices = []
				current_belongs_to_distribution_entropy = 0
				current_not_belongs_to_distribution_entropy = 0
				for table_index in current_node.ngram_fragment_row_indices:
					predictor_word = self.ngram_table[table_index][Xi]
					target_word = self.ngram_table[table_index][self.ngram_window_size-1]
					if predictor_word in Si:
						current_belongs_to_indices.append(table_index)
						try:
							current_belongs_to_distribution[target_word] += 1.0
						except KeyError:
							current_belongs_to_distribution[target_word] = 1.0
					else:
						current_not_belongs_to_indices.append(table_index)
						try:
							current_not_belongs_to_distribution[target_word] += 1.0
						except KeyError:
							current_not_belongs_to_distribution[target_word] = 1.0
				belongs_to_frequency_sum = sum(current_belongs_to_distribution.values())
				for key in current_belongs_to_distribution.keys():
					frequency = current_belongs_to_distribution[key]
					probability = frequency/belongs_to_frequency_sum
					probability_of_info_gain = probability * math.log(probability,2)
					current_belongs_to_distribution[key] = probability
					current_belongs_to_distribution_entropy += -probability_of_info_gain

				not_belongs_to_frequency_sum = sum(current_not_belongs_to_distribution.values())
				for key in current_not_belongs_to_distribution.keys():
					frequency = current_not_belongs_to_distribution[key]
					probability = frequency/not_belongs_to_frequency_sum
					probability_of_info_gain = probability * math.log(probability,2)
					current_not_belongs_to_distribution[key] = probability
					current_not_belongs_to_distribution_entropy += -probability_of_info_gain

				belongs_to_probability = (
					float(len(current_belongs_to_indices))/len(current_node.ngram_fragment_row_indices)
				)
				not_belongs_to_probability = ( 
					float(len(current_not_belongs_to_indices))/len(current_node.ngram_fragment_row_indices)
				)
				current_average_conditional_entropy = (belongs_to_probability *
					current_belongs_to_distribution_entropy + not_belongs_to_probability *
					current_not_belongs_to_distribution_entropy)
				current_reduction = current_node.entropy - current_average_conditional_entropy
				if best_question_data['reduction'] < current_reduction:
					best_question_data['reduction'] = current_reduction
					best_question_data['belongs_to_indices'] = current_belongs_to_indices
					best_question_data['belongs_to_distribution'] = current_belongs_to_distribution
					best_question_data['belongs_to_distribution_entropy'] = current_belongs_to_distribution_entropy
					best_question_data['not_belongs_to_indices'] = current_not_belongs_to_indices
					best_question_data['not_belongs_to_distribution'] = current_not_belongs_to_distribution
					best_question_data['not_belongs_to_distribution_entropy'] = current_not_belongs_to_distribution_entropy
					best_question_data['set'] = Si
					best_question_data['predictor_variable_index'] = Xi

		if best_question_data['reduction'] > self.reduction_threshold:
			current_node.set = best_question_data['set']
			current_node.predictor_variable_index = best_question_data['predictor_variable_index']
			current_node.lchild = Node()
			current_node.lchild.ngram_fragment_row_indices = best_question_data['belongs_to_indices']
			current_node.lchild.entropy = best_question_data['belongs_to_distribution_entropy']
			current_node.lchild.distribution = best_question_data['belongs_to_distribution']
			current_node.rchild = Node()
			current_node.rchild.ngram_fragment_row_indices = best_question_data['not_belongs_to_indices']
			current_node.rchild.entropy = best_question_data['not_belongs_to_distribution_entropy']
			current_node.rchild.distribution = best_question_data['not_belongs_to_distribution']
			self.node_queue.put(current_node.lchild)
			self.node_queue.put(current_node.rchild)
			current_node.lchild.parent = current_node
			current_node.rchild.parent = current_node
		else:
			logging.info("Leaf Node reached, Top Probability Distribution:%s",dict(Counter(current_node.distribution).most_common(5)))

	def question_already_asked(self, current_node, Xi, Si):

		parent = current_node.parent
		while(parent != None):
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
			set(["the","for","in","at","a"]),
		]

def init_logger():
	logger = logging.getLogger()
	handler = logging.StreamHandler()
	logging.basicConfig(filename='trsl.log', filemode='w', level=logging.INFO)
	formatter = logging.Formatter(
    	'%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
	handler.setFormatter(formatter)
	logger.addHandler(handler)
	logger.setLevel(logging.INFO)
