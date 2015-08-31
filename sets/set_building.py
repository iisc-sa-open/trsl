#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Used to build sets from the vectors provided by KMeans clustering
"""

import json
import sys

from sklearn.cluster import KMeans


def build_sets(num_words=None, num_clusters=None, vectors=None):
    """
        Used to build sets with vectors generated,
        Kmeans utilised to make the clusters of desired size
    """

    if num_words is None or num_clusters is None or vectors is None:
        return None
    word_list = []
    vector_list = []
    for index in xrange(0, num_words):
        set_data = json.loads(vectors[index])
        word_list.append(set_data[0])
        vector_list.append(set_data[1])

    kmeans_clust = KMeans(
        n_clusters=num_clusters,
        precompute_distances=True,
        n_jobs=-2, max_iter=1000,
        n_init=20
    )
    idx = kmeans_clust.fit_predict(vector_list)

    k = [[] for _ in xrange(num_clusters)]
    word_centroid_map = dict(zip(word_list, idx))
    for word in word_centroid_map.keys():
        k[word_centroid_map[word]].append(word)
    filename = sys.argv[4]
    open(
        filename, "w"
    ).write(json.dumps(k))


if __name__ == "__main__":

    NUM_WORDS = int(sys.argv[1])
    NUM_CLUSTERS = int(sys.argv[2])
    VECTORS_LIST = open(sys.argv[3], "r").read().split("\n")
    build_sets(
        num_words=NUM_WORDS, num_clusters=NUM_CLUSTERS, vectors=VECTORS_LIST
    )
