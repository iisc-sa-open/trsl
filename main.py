#! /usr/bin/env/python2
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

def trsl_operations(trsl_instance):
    """
        From pretrained trsl perform predict, tree walk operations
        as a menu driven program
    """

    if trsl_instance is None:
        logging.error("Training imcomplete")
        return

    while True:
        print "Enter your Choice:\n1> Predict next Word\n2> Random Tree Walk\n3> Exit"
        choice = int(raw_input())
        if choice == 1:
            print "Enter predictor words to predict the next:"
            print "Ten most likely words:"+str(
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
            logging.error("Error, invalid input")

def args_parser():
    """
        Initializes the format and level of the logging
    """

    trsl_instance = None
    parser = argparse.ArgumentParser(
        description='Main.py executes and trains the trsl'
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
        dest="model",
        help="Pretrained model fed to the file",
        action="store"
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        help="Config file for the model supplied here",
        action="store"
    )
    parser.add_argument(
        "-t",
        "--text",
        dest="corpus",
        help="Text corpus supplied for making the model",
        action="store"
    )
    parser.add_argument(
        "-g",
        "--group",
        dest="group",
        help="Groups of words [sets]",
        action="store"
    )
    args = parser.parse_args()
    init_logger(args)
    time.time()
    old_time = time.time()
    if args.model:
        trsl_instance = Trsl(model=args.model)
        trsl_instance.train()
    elif args.group and args.corpus and args.config:
        trsl_instance = Trsl(set_filename=args.group, corpus=args.corpus, config=args.config)
        trsl_instance.train()
    elif args.group and args.corpus:
        trsl_instance = Trsl(set_filename=args.group, corpus=args.corpus)
        trsl_instance.train()
    elif args.corpus and args.config:
        trsl_instance = Trsl(corpus=args.corpus, config=args.config)
        trsl_instance.train()
    else:
        print "Required arguments not passed, run --help for more details"
        return

    new_time = time.time()
    loading_time = new_time - old_time
    logging.info("Execution Time : "+str(loading_time))
    trsl_operations(trsl_instance)


def init_logger(args):
    """
        Initialise the logger based on user preference
    """

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

    args_parser()
