import logging
import trsl
import node
import math
import json
from collections import Counter
from matplotlib import pyplot as plt

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
tree_data = []
trsl_instance = trsl.Trsl()
trsl_instance.train()
logging.info("Loading Complete")
leaf_nodes = []
sets = []
sets_count  = [ 0 for x in range(0,101)] # numsets
xi = [ 0 for x in range(0,5)]
length_fragment_row_indices_list = []
depth_list = []

def compute_avg_entropy():

    global trsl_instance, tree_data, length_fragment_row_indices_list, depth_list

    bfs([trsl_instance.root])
    plt.xlabel("Level Avg Entropy vs Level.png")
    plt.ylabel("Avg Entropy")
    plt.plot(range(0,len(tree_data)), map(lambda x: x['avg_entropy'], tree_data),label="Probabilistic Avg Entropy")
    #plt.plot(range(0,len(tree_data)), map(lambda x: x['max_entropy'], tree_data), label="Absolute Max Entropy")
    #plt.plot(range(0,len(tree_data)), map(lambda x: x['min_entropy'], tree_data), label="Absolute Min Entropy")
    plt.legend()
    plt.savefig(
        "Avg Entropy vs Level.png"
    )
    plt.figure()
    plt.xlabel("Level")
    plt.ylabel("log2(No of Nodes)")
    plt.plot(range(0,len(tree_data)), map(lambda x: math.log(x['no_of_nodes'],2), tree_data))
    plt.savefig(
        "No of Nodes vs Level.png"
    )

    plt.figure()
    plt.xlabel("Predictor Variable index")
    plt.ylabel("No of Questions")
    plt.bar(range(0,5), xi)
    plt.savefig(
        "Xi vs no of questions.png"
    )

    plt.figure()
    plt.xlabel("Depth")
    plt.ylabel("No of Fragment Indices")
    plt.plot(depth_list, length_fragment_row_indices_list)
    plt.savefig(
        "Depth vs No of Fragment Indices"
    )

    plt.figure()
    plt.xlabel("Set index")
    plt.ylabel("No of Questions")
    plt.plot(range(0,len(sets_count)), sets_count)
    plt.savefig(
        "Set index vs no of questions.png"
    )
    open(trsl_instance.filename+".set_index","w").write(json.dumps(zip(map(list,sets),sets_count)))


def bfs(node_list):

    global depth_list, length_row_fragment_indices
    children = []
    probabilistic_average_entropy = 0
    sum_length_row_fragment_indices = 0
    probabilistic_average_entropy = sum(n.probabilistic_entropy for n in node_list)
    for node in node_list:
        sum_length_row_fragment_indices += len(node.row_fragment_indices)
        if node.rchild is not None:
            xi[node.predictor_variable_index] += 1
            try:
                sets_count[sets.index(node.set)] += 1
            except ValueError:
                sets.append(node.set)
                sets_count[len(sets) - 1] += 1
            children.append(node.rchild)
        else:
            leaf_nodes.append(node)
        if node.lchild is not None:
            children.append(node.lchild)
    length_fragment_row_indices_list.append(sum_length_row_fragment_indices / len(node_list))
    depth_list.append(node_list[0].depth)

    tree_data.append(
        {
            'avg_entropy': probabilistic_average_entropy,
            'no_of_nodes': len(node_list)
            #'max_entropy': max(node_list, key=lambda x: x.absolute_entropy).absolute_entropy,
            #'min_entropy': min(node_list, key=lambda x: x.absolute_entropy).absolute_entropy
        }
    )
    if len(children) is not 0:
        bfs(children)


compute_avg_entropy()
