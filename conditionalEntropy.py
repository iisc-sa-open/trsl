#! /usr/bin/env python2
import collections
import math

def frequencyCount(distribution):
    """
        Obtain the frequency count of the distribution
    """

    return dict(collections.Counter(distribution))


def calculateProbabilityDistribution(distribution):
    """
        Calculate Probabilty Distribution by frequency count over the Set
    """

    freqCount = frequencyCount(distribution)
    length = len(distribution)
    probabilityDistribution = {}
    for word in freqCount.keys():
        count = freqCount[word]
        probabilityDistribution[word] = {
            'count': count,
            'probability': count/length
        }
    return probabilityDistribution


def calculateEntropy(probabilitydistribution):
    """
        Calculate Entropy for the given distribution
    """

    entropy = 0
    for key in probabilitydistribution.keys():
        wordProbability = probabilitydistribution[key]['probability']
        if wordProbability != 0:
            entropy += math.log(wordProbability, 2) * wordProbability
    return -entropy
