import json
from node import Node
from question import Question

class PickleTrsl(object):

	def serialise(self, trsl_instance):

		pickled_data = {}
		pickled_data['filename'] = trsl_instance.filename
		pickled_data['root'] = str(id(trsl_instance.root))
		pickled_data['set_filename'] = trsl_instance.set_filename
		pickled_data['no_of_nodes'] = trsl_instance.no_of_nodes
		pickled_data['reduction_threshold'] = trsl_instance.reduction_threshold
		pickled_data['ngram_window_size'] = trsl_instance.ngram_window_size
		pickled_data['max_depth'] = trsl_instance.max_depth
		pickled_data['min_depth'] = trsl_instance.min_depth
		pickled_data['tree'] = {}
		pickled_data['word_sets'] = []
		for set_data in trsl_instance.word_sets:
			pickled_data['word_sets'].append(list(set_data))
		pickled_data['current_leaf_nodes'] = []
		for leaf in trsl_instance.current_leaf_nodes:
			pickled_data['current_leaf_nodes'].append(str(id(leaf)))
		tree = pickled_data['tree']
		stack = [trsl_instance.root]
		tree[str(id(trsl_instance.root))] = {}
		self.__save_data(tree, trsl_instance.root)
		tree[str(id(trsl_instance.root))]['lchild'] = None
		tree[str(id(trsl_instance.root))]['rchild'] = None
		tree[str(id(trsl_instance.root))]['parent'] = None

		while len(stack) is not 0:
			node = stack.pop()
			if node is not None:
				if not node.is_leaf():
					tree[str(id(node))]['lchild'] = str(id(node.lchild))
					tree[str(id(node.lchild))] = {'parent':str(id(node))}
					self.__save_data(tree, node.lchild)
					if not node.lchild.is_leaf():
						stack.append(node.lchild)
					else:
						tree[str(id(node.lchild))]['lchild'] = None
						tree[str(id(node.lchild))]['rchild'] = None

					tree[str(id(node))]['rchild'] = str(id(node.rchild))
					tree[str(id(node.rchild))] = {'parent':str(id(node))}
					self.__save_data(tree, node.rchild)
					if not node.rchild.is_leaf():
						stack.append(node.rchild)
					else:
						tree[str(id(node.rchild))]['rchild'] = None
						tree[str(id(node.rchild))]['lchild'] = None

		return json.dumps(pickled_data)

	def __save_data(self, tree, node):

		tree[str(id(node))]['dist'] = node.dist
		tree[str(id(node))]['absolute_entropy'] = node.absolute_entropy
		tree[str(id(node))]['probabilistic_entropy'] = node.probabilistic_entropy
		tree[str(id(node))]['depth'] = node.depth
		tree[str(id(node))]['probability'] = node.probability
		tree[str(id(node))]['parent'] = str(id(node.parent))
		tree[str(id(node))]['avg_conditional_entropy'] = node.best_question.avg_conditional_entropy
		tree[str(id(node))]['reduction'] = node.best_question.reduction
		tree[str(id(node))]['set'] = None if node.set is None else list(node.set)
		tree[str(id(node))]['predictor_variable_index'] = node.predictor_variable_index
		tree[str(id(node))]['row_fragment_indices'] = node.row_fragment_indices


	def __set_data(self, tree, temp, key):

		temp.dist = tree[key]['dist']
		temp.absolute_entropy = float(tree[key]['absolute_entropy'])
		temp.probabilistic_entropy = float(tree[key]['probabilistic_entropy'])
		temp.depth = int(tree[key]['depth'])
		temp.probability = float(tree[key]['probability'])
		temp.best_question = Question()
		temp.best_question.avg_conditional_entropy = float(tree[key]['avg_conditional_entropy'])
		temp.best_question.reduction = float(tree[key]['reduction'])
		temp.predictor_variable_index = tree[key]['predictor_variable_index']
		temp.set = None if tree[key]['set'] is None else set(tree[key]['set'])
		temp.row_fragment_indices = tree[key]['row_fragment_indices']

	def deserialise(self, trsl_instance, json_data):

		pickled_data = json.loads(json_data)
		trsl_instance.filename = pickled_data['filename']
		trsl_instance.set_filename = pickled_data['set_filename']
		trsl_instance.no_of_nodes = int(pickled_data['no_of_nodes'])
		trsl_instance.reduction_threshold = int(pickled_data['reduction_threshold'])
		trsl_instance.ngram_window_size = int(pickled_data['ngram_window_size'])
		trsl_instance.max_depth = pickled_data['max_depth']
		trsl_instance.min_depth = pickled_data['min_depth']
		trsl_instance.word_sets = []
		for set_data in pickled_data['word_sets']:
			trsl_instance.word_sets.append(set(set_data))
		trsl_instance.root = Node(trsl_instance.ngram_window_size)
		tree = pickled_data['tree']

		stack = [pickled_data['root']]
		nodes = {pickled_data['root']:trsl_instance.root}
		while len(stack) is not 0:

			key = stack.pop()
			temp = nodes[key]
			self.__set_data(tree, temp, key)
			temp.parent = None if tree[key]['parent'] is None else nodes[tree[key]['parent']]

			if  tree[key]['lchild'] is not None or tree[key]['rchild'] is not None:
				nodes[str(tree[key]['lchild'])] = Node(trsl_instance.ngram_window_size)
				nodes[str(tree[key]['rchild'])] = Node(trsl_instance.ngram_window_size)
				temp.lchild = nodes[tree[key]['lchild']]
				temp.rchild = nodes[tree[key]['rchild']]
				stack.append(tree[key]['lchild'])
				stack.append(tree[key]['rchild'])
			else:
				temp.lchild = None
				temp.rchild = None

		trsl_instance.current_leaf_nodes = []
		for leaf in pickled_data['current_leaf_nodes']:
			trsl_instance.current_leaf_nodes.append(nodes[str(leaf)])
