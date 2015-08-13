#! /usr/bin/env/python2
"""
    Validates the trsl instance after training it.
    Constraints verified:
    * There is always reduction in a node from the parent
    * Correct probability distribution from the parent node between the children
"""

import sys
import os
import inspect
import argparse
import time
import unittest

CURRENT_DIR = os.path.dirname(
    os.path.abspath(
        inspect.getfile(inspect.currentframe())
    )
)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import trsl
from trsl import Trsl

class TrslTestCase(unittest.TestCase):
    """Tests for `trsl.py`."""

    def setUp(self):
        """Create instance of trsl and train the instance for each test"""

        self.trsl_instance = trsl.Trsl()
        self.trsl_instance.train()

    @staticmethod
    def bfs(node_list, validate_condition):

        children = []
        for node in node_list:
            if node.rchild is not None:
                validate_condition(node)
                children.append(node.rchild)
                children.append(node.lchild)

        if len(children) is not 0:
            TrslTestCase.bfs(children, validate_condition)

    def test_reduction(self):
        """Validate significant reduction of each node from parent node"""

        TrslTestCase.bfs(
            [self.trsl_instance.root],
            lambda node: self.assertGreater(
                node.best_question.reduction,
                0,
                msg="Reduction less than Zero"
            )
        )

    def test_probability_division(self):
        """Validate correct probability distribution from parent between the children"""

        TrslTestCase.bfs(
            [self.trsl_instance.root],
            lambda node: self.assertAlmostEqual(
                node.rchild.probability + node.lchild.probability,
                node.probability,
                places=4,
                msg="Correct probability division failed across children"
            )
        )


if __name__ == "__main__":

    unittest.main()
