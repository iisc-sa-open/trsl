#! /usr/bin/env/python2

import os
import sys
import inspect
import time
from collections import Counter


CURRENT_DIR = os.path.dirname(
    os.path.abspath(
        inspect.getfile(inspect.currentframe())
    )
)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
import trsl

ngram_window_size = 6
trsl.init_logger()
time.time()
old = time.time()
trsl_instance = trsl.Trsl(ngram_window_size)
trsl_instance.train("../last_question")
new = time.time()
trsl.logging.info("Execution Time : "+str(new-old))
while(True):
	print("\nEnter %s words to predict the next one:"%(ngram_window_size-1))
	print("Ten most likely words:"+str(Counter(trsl_instance.predict(raw_input().lower().split())).most_common(10)))
