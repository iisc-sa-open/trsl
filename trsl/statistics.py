#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Used for producing graphs for trsl
"""

import argparse
import json
import logging
import math
from trsl import Trsl
from collections import Counter
from matplotlib import pyplot as plt
from nltk.tokenize import RegexpTokenizer


def args_parser():
    """
        Used for command line argument parsing
    """

    parser = argparse.ArgumentParser(
        description='script used for generating graphs over precomputed model'
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        help="Pretrained model fed to the file",
        action="store",
        required=True
    )
    parser.add_argument(
        "-n",
        "--ngram",
        dest="ngram",
        help="Ngram Size",
        action="store",
        required=True
    )
    parser.add_argument(
        "-c",
        "--clusters",
        dest="clusters",
        help="Cluster size input",
        action="store",
        required=True
    )
    args = parser.parse_args()
    if args.model and args.clusters and args.ngram:
        trsl_instance = Trsl(model=args.model)
        return trsl_instance, int(args.clusters), int(args.ngram)
    else:
        print 'Required arguments not passed, model, cluster, ngram'


def init_logging():
    """
        Initialise logging for statistics.py
    """

    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def plot_graphs():
    """
        Plotting all graphs
    """

    global trsl_instance, tree_data, length_fragment_row_indices_list
    global depth_list, sets_count, sets, xi

    bfs([trsl_instance.root])
    plt.xlabel("Level Avg Entropy vs Level.png")
    plt.ylabel("Avg Entropy")
    plt.plot(
        xrange(0, len(tree_data)),
        map(lambda x: x['avg_entropy'], tree_data),
        label="Probabilistic Avg Entropy"
    )
    plt.legend()
    plt.savefig(
        "Avg Entropy vs Level.png"
    )
    plt.figure()
    plt.xlabel("Level")
    plt.ylabel("log2(No of Nodes)")
    plt.plot(
        xrange(0, len(tree_data)),
        map(lambda x: math.log(x['no_of_nodes'], 2), tree_data)
    )
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
    open(
        trsl_instance.filename + ".set_index", "w"
    ).write(json.dumps(zip(map(list, sets), sets_count)))

    plt.figure()
    tokenizer = RegexpTokenizer(r'(\w+(\'\w+)?)|\.')
    common_words = Counter(
        tokenizer.tokenize(
            open(
                trsl_instance.filename,
                "r"
            ).read().lower()
        )
    )
    sets_avg_freq = []

    for s in sets:
        temp_list = []
        for word in s:
            temp_list.append(common_words[word])
        temp_list.sort()
        sets_avg_freq.append(temp_list[len(temp_list) / 2])
    plt.xlabel("Median frequency of words in set")
    plt.ylabel("No of Questions")
    plt.bar(sets_avg_freq, sets_count[:len(sets)])
    plt.savefig(
        "Median Freq of set vs no of questions.png"
    )

    plt.figure()
    plt.xlabel("Predictor Variable index")
    plt.ylabel("No of Questions")
    plt.bar(range(0, len(xi)), xi)
    plt.savefig(
        "Xi vs no of questions.png"
    )


def bfs(node_list):
    """
        BFS traversal through all the nodes
    """

    global depth_list, length_row_fragment_indices, trsl_instance
    children = []
    sum_length_row_fragment_indices = 0
    probabilistic_average_entropy = sum(
        n.probabilistic_entropy for n in node_list
    )
    for node in node_list:
        sum_length_row_fragment_indices += node.len_data_fragment
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
    length_fragment_row_indices_list.append(
        sum_length_row_fragment_indices / len(node_list)
    )
    depth_list.append(node_list[0].depth)

    tree_data.append(
        {
            'avg_entropy': probabilistic_average_entropy,
            'no_of_nodes': len(node_list)
        }
    )
    if len(children) is not 0:
        bfs(children)


if __name__ == "__main__":

    init_logging()
    trsl_instance, clusters, ngram = args_parser()
    tree_data = []
    logging.info("Loading Complete")
    leaf_nodes = []
    sets = []
    sets_count = [0 for x in range(0, clusters)]
    xi = [0 for x in range(0, ngram)]
    length_fragment_row_indices_list = []
    depth_list = []

    if trsl_instance is None:
        logging.error("Error, trsl not trained from precomputed data")
    else:
        plot_graphs()
