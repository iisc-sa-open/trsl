#! /usr/bin/env/python2
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
    set_filename = open("google-10000-vectors.json", "r").read().split("\n")
    word_list = []
    vector_list = []
    for index in range(0, num_words):
        set_data = json.loads(set_filename[index])
        word_list.append(set_data[0])
        vector_list.append(set_data[1])

    old_time = time.time()
    kmeans_clust = KMeans(n_clusters=num_clusters)
    idx = kmeans_clust.fit_predict(vector_list)
    new_time = time.time()
    print("Time taken for clustering", new_time - old_time)

    k = [[] for x in range(num_clusters)]
    word_centroid_map = dict(zip(word_list, idx))
    for word in word_centroid_map.keys():
        k[word_centroid_map[word]].append(word)
    filename = "Kmeans-"+str(num_words)+"words-"+str(num_clusters)+"clusters.json"
    open(
        filename, "w"
    ).write(json.dumps(k))
