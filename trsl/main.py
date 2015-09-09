#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    main.py is used for training a trsl instance based on specified corpus,
    pre-trained model.
"""


import argparse
import time
import logging
from collections import Counter
from trsl import Trsl


def trsl_operations(trsl_instance, logger):
    """
        From pretrained trsl perform predict, tree walk operations
        as a menu driven program
    """

    if trsl_instance is None:
        logger.error("Training imcomplete")
        return

    while True:
        print """Enter your Choice:
            1> Predict next Word
            2> Random Tree Walk
            3> Exit"""
        choice = int(raw_input())
        if choice == 1:
            print "Enter predictor words to predict the next:"
            print "Ten most likely words:" + str(
                Counter(
                    trsl_instance.predict(
                        raw_input().lower().split()
                    )
                ).most_common(10))
        elif choice == 2:
            print "Enter the no of words to be generated: "
            no_of_words = int(raw_input())
            print "Enter predictor words to predict the next: "
            print "Text generated is :\n" + " ".join(
                trsl_instance.tree_walk(
                    raw_input().strip().split(),
                    no_of_words
                )
            )
        elif choice == 3:
            return
        else:
            logger.error("Error, invalid input")


def args_parser():
    """
        Initializes the format and level of the logging
    """

    trsl_instance = None
    parser = argparse.ArgumentParser(
        description='Script used to generate models'
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
        dest="model",
        help="pre-computed model file path",
        action="store"
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        help="config file for the model generation",
        action="store"
    )
    parser.add_argument(
        "-t",
        "--text",
        dest="corpus",
        help="text corpus for model generation",
        action="store"
    )
    parser.add_argument(
        "-g",
        "--group",
        dest="group",
        help="groups of words, preclustered words based on vectors [ sets ]",
        action="store"
    )
    args = parser.parse_args()
    logger = init_logger(args)
    time.time()
    old_time = time.time()
    # Load a precomputed model for trsl
    if args.model:
        trsl_instance = Trsl(model=args.model)
    # Build trsl using precomputed word sets and a config file
    elif args.group and args.corpus and args.config:
        trsl_instance = Trsl(
            set_filename=args.group, corpus=args.corpus, config=args.config
        )
    # Build trsl using precomputed word sets
    elif args.group and args.corpus:
        trsl_instance = Trsl(set_filename=args.group, corpus=args.corpus)
    # Build trsl and word sets from only the corpus and config file
    elif args.corpus and args.config:
        trsl_instance = Trsl(corpus=args.corpus, config=args.config)
    # Insufficient arguments passed to build trsl
    else:
        print "Required arguments not passed, run --help for more details"
        return

    new_time = time.time()
    loading_time = new_time - old_time
    logger.info("Execution Time : " + str(loading_time))
    trsl_operations(trsl_instance, logger)


def init_logger(args):
    """
        Initialise the logger based on user preference
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

    args_parser()
