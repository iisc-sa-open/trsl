#! /usr/bin/env/python2
"""
    A test writen with the same data and parameters used
    to validate with the manual calculations present in
    last_question_hand_calcs.ods

    The sets returned from Trsl.build_sets have also been
    stubbed to be the same as the one used for the manual calculations.
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
    #trsl_instance.tree_walk(["It", "was", "now", "some" ,"time"], 10)
    while True:
        print "\nEnter predictor words to predict the next:"
        print "Ten most likely words:"+str(
            Counter(
                trsl_instance.predict(
                    raw_input().lower().split()
                )
            ).most_common(10))
