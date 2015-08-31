#! /usr/bin/env/python2
"""
    Validates the trsl instance after training it.
    Constraints verified:
    * There is always reduction in a node from the parent
    * Correct probability distribution from the parent node
        between the children
    * Validate len_data_fragment of every node is less than
        input sample size threshold
    * Validate every leaf node contains a word probability
"""

import argparse
import inspect
import logging
import os
import sys
import unittest

import trsl


CURRENT_DIR = os.path.dirname(
    os.path.abspath(
        inspect.getfile(inspect.currentframe())
    )
)
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)


class TrslTestCase(unittest.TestCase):
    """Tests for `trsl.py`."""

    def setUp(self):
        """Create instance of trsl and train the instance for each test"""

        global MODEL
        logger = logging.getLogger('Trsl')
        logger.setLevel(logging.ERROR)
        self.trsl_instance = trsl.Trsl(model=MODEL)

    @staticmethod
    def bfs(node_list, validate_condition):

        children = []
        for node in node_list:
            # if it is an internal node
            if node.rchild is not None:
                # Validates conditions specified by individual tests,
                # on the current node
                validate_condition(node)
                # push the children of the current node into the stack
                children.append(node.rchild)
                children.append(node.lchild)

        if len(children) is not 0:
            TrslTestCase.bfs(children, validate_condition)

    def test_reduction(self):
        """
            Validate significant reduction of each node from parent node
        """

        TrslTestCase.bfs(
            [self.trsl_instance.root],
            lambda node: self.assertGreater(
                node.best_question.reduction,
                0,
                msg="Reduction less than Zero"
            )
        )

    def test_probability_division(self):
        """
            Validate correct probability distribution
            from parent between the children
        """

        TrslTestCase.bfs(
            [self.trsl_instance.root],
            lambda node: self.assertAlmostEqual(
                node.rchild.probability + node.lchild.probability,
                node.probability,
                places=4,
                msg="Correct probability division failed across children"
            )
        )

    def test_sample_size(self):
        """
            Validate if data fragment size,
            for every node is greater than sample size
        """

        TrslTestCase.bfs(
            [self.trsl_instance.root],
            lambda node: self.assertGreater(
                node.len_data_fragment,
                self.trsl_instance.sample_size,
                msg="""
                    Sample size should be greater than
                    data fragment length for every node
                """
            )
        )

    def test_leaf_word_probability(self):
        """Validate if every leaf node has a word probability"""

        for leaf_node in self.trsl_instance.current_leaf_nodes:
            self.assertIsNotNone(
                leaf_node.word_probability,
                msg="Leaf node does not have word probability"
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Test script for trsl.py'
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Model file path",
        action="store",
        required=True
    )
    args = parser.parse_args()
    if args.model:
        MODEL = args.model
        runner = unittest.TextTestRunner()
        itersuite = unittest.TestLoader().loadTestsFromTestCase(TrslTestCase)
        runner.run(itersuite)
