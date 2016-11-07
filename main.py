import util
import numpy as np
import logging
import itertools

if __name__ == "__main__":
    np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Program started")

    learning_examples = [l for l in util.unique(itertools.permutations('00000001'))]
    learning_examples = np.array(learning_examples, dtype=int)
    logger.debug("Learning Examples : \n{}".format(learning_examples))

    np.random.seed(0)
    w1 = np.random.uniform(-1, 1, (3, 8))
    b1 = np.random.uniform(-1, 1, (3, 1))
    w2 = np.random.uniform(-1, 1, (8, 3))
    b2 = np.random.uniform(-1, 1, (8, 1))

    iterations = 1000
    learning_rate = 0.3

    for i in range(iterations):
        error_sum = 0
        for j in learning_examples:
            a1 = np.array([j]).T
            # logger.debug("Layer 1 Activation : \n{}".format(a1))
            logger.debug("In : {}".format(a1.T))

            a2 = (w1@a1)+b1
            # logger.debug("Layer 2 Activation : \n{}".format(util.sigmoid(a2)))

            a3 = (w2@a2)+b2
            # logger.debug("Layer 3 Activation : \n{}".format(a3))
            logger.debug("Out : {}\n".format(a3.T))

            d3 = a3 * (util.sigmoid(a3) - a1)
            d3 = a1-a3
            # logger.debug("Layer 3 Delta: \n{}".format(d3))

            d2 = util.dsigmoid(a2)*(w2.T@d3)
            # logger.debug("Layer 2 Delta: \n{}".format(d2))

            w2delta = learning_rate * (np.outer(d3, util.sigmoid(a2)))
            w2 += w2delta
            # logger.debug("Weights 2: \n{}".format(w2))
            # logger.debug("Weights 2 Delta: \n{}".format(w2delta))

            w1delta = learning_rate * (np.outer(d2, util.sigmoid(a1)))
            w1 += w1delta
            # logger.debug("Weights 1: \n{}".format(w1))
            # logger.debug("Weights 1 Delta: \n{}".format(w1delta))

            b2 += learning_rate * d3
            # logger.debug("Layer 2 Bias : \n{}".format(b2))
            # logger.debug("Layer 2 Bias Delta : \n{}".format(learning_rate * d3))

            b1 += learning_rate * d2
            # logger.debug("Layer 1 Bias : \n{}".format(b1))
            # logger.debug("Layer 2 Bias Delta : \n{}".format(learning_rate * d2))

            error_sum += abs(np.sum(d3))

        logger.debug("Overall error after {} iterations : {}".format(i+1, error_sum))
        epsilon = 0.00000001
        if abs(np.sum(d3)) < epsilon:
            break

    logger.info("Final error after {} iterations : {}".format(iterations, error_sum))









