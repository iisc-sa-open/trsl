#! /usr/bin/env/python2
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    Example to create and load a trsl instance, train the same
    and perform a random tree walk with the trsl instance
"""


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

def init_logger_parser():
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
    parser.add_argument(
        "-m",
        "--model",
        help="Model file path",
        action="store",
        required=True
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
    return args.model

if __name__ == "__main__":

    model = init_logger_parser()
    time.time()
    OLD_TIME = time.time()
    if model is not None:
        trsl_instance = trsl.Trsl(model=model)
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
    else:
        logging.error("Pre trained model path needs to be passed")
