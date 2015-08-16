#! /usr/bin/env/python2
# -*- coding: utf-8 -*-
# Copyright of the Indian Institute of Science's Speech and Audio group.

"""
    main.py is used for training a trsl instance based on specified corpus,
    pre-trained model.
"""


import argparse
import inspect
import os
import sys
import time
import logging
from collections import Counter
from trsl import Trsl

def args_parser():
    """
        Initializes the format and level of the logging
    """

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
    args = parser.parse_args()
    init_logger(args)
    time.time()
    OLD_TIME = time.time()
    if args.model:
        trsl_instance = Trsl(model=args.model)
        trsl_instance.train()
    elif args.config:
        trsl_instance = Trsl(config=args.config)
        trsl_instance.train()
    elif args.corpus:
        trsl_instance = Trsl(corpus=args.corpus)
        trsl_instance.train()
    else:
        print "Required arguments not passed, run --help for more details"
        return

    NEW_TIME = time.time()
    logging.info("Execution Time : "+str(NEW_TIME - OLD_TIME))

    print("Predict -> Enter a sentence: ")
    print(
            trsl_instance.predict(
                raw_input().split()
            )
    )

def init_logger(args):

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
