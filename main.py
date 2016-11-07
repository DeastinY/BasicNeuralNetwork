import math
import util
import numpy as np
import logging
import itertools


def g(z):
    return 1/(1+math.exp(-z))

if __name__ == "__main__":
    np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Program started")

    learning_examples = [l for l in util.unique(itertools.permutations('00000001'))]
    learning_examples = np.array(learning_examples, dtype=int)
    logger.debug("Learning Examples : \n{}".format(learning_examples))

    blearn = np.r_[np.full((1, 8), 1, dtype=int), learning_examples]

    np.random.seed(0)
    w1 = 2*np.random.random((3, 8)) - 1
    bw1 = np.c_[np.full((3, 1), 1, dtype=int), w1]
    w2 = 2*np.random.random((8, 3)) - 1
    bw2 = np.c_[np.full((8, 1), 1, dtype=int), w2]

    dw1 = np.zeros((3, 8))
    dw2 = np.zeros((8, 3))

    iterations = 50
    learning_rate = 0.5

    for i in range(iterations):
        a1 = learning_examples
        ba1 = blearn
        logger.debug("Layer 1 Activation : \n{}".format(learning_examples))
        a2 = util.sigmoid(bw1@ba1)
        logger.debug("Layer 2 Activation : \n{}".format(a2))
        ba2 = np.r_[np.full((1, 8), 1, dtype=int), a2]
        a3 = bw2@ba2
        logger.debug("Layer 3 Activation : \n{}".format(a3))
        ba3 = np.r_[np.full((1, 8), 1, dtype=int), a3]

        d3 = a3-a1

        logger.debug("Layer 3 Delta: \n{}".format(d3))
        d2 = w2.T@d3*(a2 * (1 - a2))
        logger.debug("Layer 2 Delta: \n{}".format(d2))
        d1 = w1.T@d2*(a1 * (1 - a1))
        logger.debug("Layer 1 Delta: \n{}".format(d3))

        dw2 += (d3 @ a2.T)
        logger.debug("Delta Weights 2: \n{}".format(w2))
        w2 = 1/iterations*(dw2+learning_rate*w2)
        bw2 = np.c_[np.full((8, 1), 1/iterations, dtype=int), w2]

        dw1 += (d2 @ a1.T)
        logger.debug("Delta Weights 1: \n{}".format(w1))
        w1 = 1/iterations*(dw1+learning_rate*w1)
        bw1 = np.c_[np.full((3, 1), 1/iterations, dtype=int), w1]

    logger.info("Output after {} iterations : \n{}".format(iterations, a3))









