#! /usr/bin/env/python2

import os
import sys
import inspect
import time


CURRENT_DIR = os.path.dirname(
    os.path.abspath(
        inspect.getfile(inspect.currentframe())
    )
)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
import trsl


if __name__ == '__main__':

	trsl.init_logger()
	time.time()
	old = time.time()
	trsl_instance = trsl.Trsl(6)
	trsl_instance.train("../last_question")
	new = time.time()
	trsl.logging.info("Execution Time : "+str(new-old))
