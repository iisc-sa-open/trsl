import json

def serialise(trsl_instance):

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
	tree[id(trsl_instance.root)]['entropy'] = trsl_instance.root.entropy
	while len(stack) is not 0:
		node = stack.pop()
		if node is not None:
			if node.lchild is not None:
				stack.append(node.lchild)
				tree[id(node)]['lchild'] = id(node.lchild)
				tree[id(node.lchild)] = {}
				if node.lchild.set is not None:
					tree[id(node.lchild)]['set'] = list(node.lchild.set)
					tree[id(node.lchild)]['predictor_variable_index'] = node.lchild.predictor_variable_index
				tree[id(node.lchild)]['dist'] = node.lchild.dist
				tree[id(node.lchild)]['entropy'] = node.lchild.entropy

			if node.rchild is not None:
				stack.append(node.rchild)
				tree[id(node)]['rchild'] = id(node.rchild)
				tree[id(node.rchild)] = {}
				if node.rchild.set is not None:
					tree[id(node.rchild)]['set'] = list(node.rchild.set)
					tree[id(node.rchild)]['predictor_variable_index'] = node.rchild.predictor_variable_index
				tree[id(node.rchild)]['dist'] = node.rchild.dist
				tree[id(node.rchild)]['entropy'] = node.rchild.entropy
	
	return json.dumps(pickled_data)

def deserialise(trsl_instance, json_data):

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
			temp.entropy = float(tree[key]['entropy'])
			temp.set = set(tree[key]['set'])
			temp.predictor_variable_index = int(tree[key]['predictor_variable_index'])
			nodes[str(tree[key]['lchild'])] = Node()
			nodes[str(tree[key]['rchild'])] = Node()
			temp.lchild = nodes[str(tree[key]['lchild'])]
			temp.rchild = nodes[str(tree[key]['rchild'])]
			stack.append(str(tree[key]['lchild']))
			stack.append(str(tree[key]['rchild']))
		except KeyError:
			continue