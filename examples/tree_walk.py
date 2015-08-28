#! /usr/bin/env/python2
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Example for prediction of the target variable based on the provided
    predictor variables using trsl instance
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


def init_parser():
    """
        Initializes the format and level of the logging
    """
    parser = argparse.ArgumentParser(
        description="""Example script for target word prediction
            based on the input predictor variables of ngram window size"""
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="increase output verbosity",
        action="store_true"
    )
    parser.add_argument(
        "-s",
        "--silent",
        help="silence all logging",
        action="store_true"
    )
    parser.add_argument(
        "-m",
        "--model",
        help="pre-computed model file path",
        action="store",
        required=True
    )
    args = parser.parse_args()
    model = args.model
    logger = init_logger(args)
    if model is not None:
        OLD_TIME = time.time()
        trsl_instance = trsl.Trsl(model=model)
        NEW_TIME = time.time()
        logger.info("Execution Time : " + str(NEW_TIME - OLD_TIME))
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
    else:
        logger.error("Pre trained model path needs to be passed")

def init_logger(args):
    """
        Initialise the logger based on given conditions
    """

    logger = logging.getLogger('Trsl')
    if args.silent:
        logger.setLevel(logging.ERROR)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    return logger

if __name__ == "__main__":

    init_parser()