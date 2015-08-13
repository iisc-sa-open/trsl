#! /usr/bin/env/python2
"""
    Example to create and load a trsl instance, train the same
    and perform a random tree walk with the trsl instance
"""


from collections import Counter
import argparse
import inspect
import os
import sys
import time
import logging


CURRENT_DIR = os.path.dirname(
    os.path.abspath(
        inspect.getfile(inspect.currentframe())
    )
)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import trsl

def init_logger():
    """
        Initializes the format and level of the logging
    """
    parser = argparse.ArgumentParser(
        description='Test script for trsl.py'
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Increase output verbosity",
        action="store_true"
    )
    parser.add_argument(
        "-s",
        "--silent",
        help="Silence all logging",
        action="store_true"
    )
    args = parser.parse_args()
    if args.silent:
        return
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

if __name__ == "__main__":

    init_logger()
    time.time()
    OLD_TIME = time.time()
    trsl_instance = trsl.Trsl()
    trsl_instance.train()
    NEW_TIME = time.time()
    trsl.logging.info("Execution Time : "+str(NEW_TIME - OLD_TIME))

    while True:
        print "\nEnter the no of words to be generated: "
        no_of_words = int(raw_input())
        print "\nEnter predictor words to predict the next: "
        print "Text generated is :\n" + " ".join(
            trsl_instance.tree_walk(
                raw_input().strip().split(),
                no_of_words
            )
        )
