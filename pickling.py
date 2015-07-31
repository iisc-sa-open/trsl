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

		tree = pickled_data['tree']
		stack = [trsl_instance.root]
		tree[id(trsl_instance.root)] = {}
		tree[id(trsl_instance.root)]['set'] = list(trsl_instance.root.set)
		tree[id(trsl_instance.root)]['dist'] = trsl_instance.root.dist
		tree[id(trsl_instance.root)]['predictor_variable_index'] = trsl_instance.root.predictor_variable_index
		tree[id(trsl_instance.root)]['absolute_entropy'] = trsl_instance.root.absolute_entropy
		tree[id(trsl_instance.root)]['probabilistic_entropy'] = trsl_instance.root.probabilistic_entropy
		tree[id(trsl_instance.root)]['depth'] = trsl_instance.root.depth
		tree[id(trsl_instance.root)]['probability'] = trsl_instance.root.probability
		tree[id(trsl_instance.root)]['parent'] = None
		tree[id(trsl_instance.root)]['avg_conditional_entropy'] = trsl_instance.root.best_question.avg_conditional_entropy
		tree[id(trsl_instance.root)]['reduction'] = trsl_instance.root.best_question.reduction
		while len(stack) is not 0:
			node = stack.pop()
			if node is not None:
				if node.lchild is not None:
					stack.append(node.lchild)
					tree[id(node)]['lchild'] = str(id(node.lchild))
					tree[id(node.lchild)] = {}
					if node.lchild.set is not None:
						tree[id(node.lchild)]['set'] = list(node.lchild.set)
						tree[id(node.lchild)]['predictor_variable_index'] = node.lchild.predictor_variable_index
					tree[id(node.lchild)]['dist'] = node.lchild.dist
					tree[id(node.lchild)]['absolute_entropy'] = node.lchild.absolute_entropy
					tree[id(node.lchild)]['probabilistic_entropy'] = node.lchild.probabilistic_entropy
					tree[id(node.lchild)]['depth'] = node.lchild.depth
					tree[id(node.lchild)]['probability'] = node.lchild.probability
					tree[id(node.lchild)]['parent'] = str(id(node))
					tree[id(node.lchild)]['avg_conditional_entropy'] = node.lchild.best_question.avg_conditional_entropy
					tree[id(node.lchild)]['reduction'] = node.lchild.best_question.reduction

				if node.rchild is not None:
					stack.append(node.rchild)
					tree[id(node)]['rchild'] = str(id(node.rchild))
					tree[id(node.rchild)] = {}
					if node.rchild.set is not None:
						tree[id(node.rchild)]['set'] = list(node.rchild.set)
						tree[id(node.rchild)]['predictor_variable_index'] = node.rchild.predictor_variable_index
					tree[id(node.rchild)]['dist'] = node.rchild.dist
					tree[id(node.rchild)]['absolute_entropy'] = node.rchild.absolute_entropy
					tree[id(node.rchild)]['probabilistic_entropy'] = node.rchild.probabilistic_entropy
					tree[id(node.rchild)]['depth'] = node.rchild.depth
					tree[id(node.rchild)]['probability'] = node.rchild.probability
					tree[id(node.rchild)]['parent'] = str(id(node))
					tree[id(node.rchild)]['avg_conditional_entropy'] = node.rchild.best_question.avg_conditional_entropy
					tree[id(node.rchild)]['reduction'] = node.rchild.best_question.reduction
		
		return json.dumps(pickled_data)

	def deserialise(self, trsl_instance, json_data):

		pickled_data = json.loads(json_data)
		trsl_instance.filename = pickled_data['filename']
		trsl_instance.set_filename = pickled_data['set_filename']
		trsl_instance.no_of_nodes = int(pickled_data['no_of_nodes'])
		trsl_instance.reduction_threshold = int(pickled_data['reduction_threshold'])
		trsl_instance.ngram_window_size = int(pickled_data['ngram_window_size'])
		trsl_instance.max_depth = pickled_data['max_depth']
		trsl_instance.min_depth = pickled_data['min_depth']
		trsl_instance.root = Node()
		tree = pickled_data['tree']

		stack = [pickled_data['root']]
		nodes = {pickled_data['root']:trsl_instance.root}
		while len(stack) is not 0:

			key = stack.pop()
			temp = nodes[key]
			try:
				temp.dist = tree[key]['dist']
				temp.predictor_variable_index = int(tree[key]['predictor_variable_index'])
				temp.absolute_entropy = float(tree[key]['absolute_entropy'])
				temp.probabilistic_entropy = float(tree[key]['probabilistic_entropy'])
				temp.depth = int(tree[key]['depth'])
				temp.probability = float(tree[key]['probability'])
				temp.best_question = Question()
				temp.best_question.avg_conditional_entropy = float(tree[key]['avg_conditional_entropy'])
				temp.best_question.reduction = float(tree[key]['reduction'])
				if tree[key]['parent'] is None:
					temp.parent = None
				else:
					temp.parent = nodes[tree[key]['parent']]
				temp.set = set(tree[key]['set'])
				nodes[str(tree[key]['lchild'])] = Node()
				nodes[str(tree[key]['rchild'])] = Node()
				temp.lchild = nodes[str(tree[key]['lchild'])]
				temp.rchild = nodes[str(tree[key]['rchild'])]
				stack.append(tree[key]['lchild'])
				stack.append(tree[key]['rchild'])
			except KeyError:
				continue