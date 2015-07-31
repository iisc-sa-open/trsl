import logging
import trsl
import node
import math
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

def compute_avg_entropy():

    global trsl_instance

    bfs([trsl_instance.root])
    plt.xlabel("Level")
    plt.ylabel("Avg Entropy")
    plt.plot(range(0,len(tree_data)), map(lambda x: x['avg_entropy'], tree_data),label="Probabilistic Avg Entropy")
    plt.plot(range(0,len(tree_data)), map(lambda x: x['max_entropy'], tree_data), label="Absolute Max Entropy")
    plt.plot(range(0,len(tree_data)), map(lambda x: x['min_entropy'], tree_data), label="Absolute Min Entropy")
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



def bfs(node_list):

    children = []
    probabilistic_average_entropy = 0
    probabilistic_average_entropy = sum(n.probabilistic_entropy for n in node_list)

    for node in node_list:
        if node.rchild is not None:
            children.append(node.rchild)
        if node.lchild is not None:
            children.append(node.lchild)
    tree_data.append(
        {
            'avg_entropy': probabilistic_average_entropy,
            'no_of_nodes': len(node_list),
            'max_entropy': max(node_list, key=lambda x: x.absolute_entropy).absolute_entropy,
            'min_entropy': min(node_list, key=lambda x: x.absolute_entropy).absolute_entropy
        }
    )
    if len(children) is not 0:
        bfs(children)


compute_avg_entropy()
