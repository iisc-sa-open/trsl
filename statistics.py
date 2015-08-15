#! /usr/bin/env/python2
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

import logging
import trsl
import node
import math
import json
from collections import Counter
from matplotlib import pyplot as plt
from nltk.tokenize import RegexpTokenizer
import code

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
sets_count = [0 for x in range(0, 500)] # numsets
xi = [0 for x in range(0, 10)]
length_fragment_row_indices_list = []
depth_list = []

def compute_avg_entropy():

    global trsl_instance, tree_data, length_fragment_row_indices_list, depth_list, sets_count, sets, xi

    bfs([trsl_instance.root])
    plt.xlabel("Level Avg Entropy vs Level.png")
    plt.ylabel("Avg Entropy")
    plt.plot(xrange(0, len(tree_data)), map(lambda x: x['avg_entropy'], tree_data), label="Probabilistic Avg Entropy")
    plt.legend()
    plt.savefig(
        "Avg Entropy vs Level.png"
    )
    plt.figure()
    plt.xlabel("Level")
    plt.ylabel("log2(No of Nodes)")
    plt.plot(xrange(0, len(tree_data)), map(lambda x: math.log(x['no_of_nodes'], 2), tree_data))
    plt.savefig(
        "No of Nodes vs Level.png"
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
    plt.bar(xrange(0, len(sets_count)), sets_count)
    plt.savefig(
        "Set index vs no of questions.png"
    )
    open(trsl_instance.filename+".set_index", "w").write(json.dumps(zip(map(list, sets), sets_count)))

    plt.figure()
    tokenizer = RegexpTokenizer(r'(\w+(\'\w+)?)|\.')
    common_words = Counter(tokenizer.tokenize(open(trsl_instance.filename, "r").read().lower()))
    sets_avg_freq = []

    for s in sets:
        temp_list = []
        for word in s:
            temp_list.append(common_words[word])
        temp_list.sort()
        sets_avg_freq.append(temp_list[len(temp_list)/2])
    plt.xlabel("Median frequency of words in set")
    plt.ylabel("No of Questions")
    plt.bar(sets_avg_freq, sets_count[:len(sets)])
    plt.savefig(
        "Median Freq of set vs no of questions.png"
    )


    plt.figure()
    plt.xlabel("Predictor Variable index")
    plt.ylabel("No of Questions")
    plt.bar(range(0, 10), xi)
    plt.savefig(
        "Xi vs no of questions.png"
    )

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
        }
    )
    if len(children) is not 0:
        bfs(children)


compute_avg_entropy()
