#! /usr/bin/env/python2
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Used to build sets
"""

import sys
import json
import time
from sklearn.cluster import KMeans

if __name__ == "__main__":

    num_words = int(sys.argv[1])
    num_clusters = int(sys.argv[2])
    set_filename = open(sys.argv[3], "r").read().split("\n")
    word_list = []
    vector_list = []
    for index in xrange(0, num_words):
        set_data = json.loads(set_filename[index])
        word_list.append(set_data[0])
        vector_list.append(set_data[1])

    kmeans_clust = KMeans(n_clusters=num_clusters, precompute_distances=True, n_jobs=-2, max_iter=1000, n_init=20)
    idx = kmeans_clust.fit_predict(vector_list)

    k = [[] for x in xrange(num_clusters)]
    word_centroid_map = dict(zip(word_list, idx))
    for word in word_centroid_map.keys():
        k[word_centroid_map[word]].append(word)
    filename = sys.argv[4]
    open(
        filename, "w"
    ).write(json.dumps(k))