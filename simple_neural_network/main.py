import util
import numpy as np
import logging
import random
import itertools

np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info("Program started")

def neural_net(iterations, learning_rate, random_weights = True, shuffle_samples = True):
    learning_examples = [l for l in util.unique(itertools.permutations('00000001'))]
    learning_examples = np.array(learning_examples, dtype=int)
    logger.debug("Learning Examples : \n{}".format(learning_examples))

    if not random_weights:
        np.random.seed(0)
    w1 = np.random.uniform(-1, 1, (3, 8))
    b1 = np.random.uniform(-1, 1, (3, 1))
    w2 = np.random.uniform(-1, 1, (8, 3))
    b2 = np.random.uniform(-1, 1, (8, 1))

    errors = []

    for i in range(iterations):
        error_sum = 0
        index=list(range(8))
        if shuffle_samples:
            np.random.shuffle(index)
        for j in index:
            a1 = np.array([learning_examples[j]]).T
            # logger.debug("Layer 1 Activation : \n{}".format(a1))

            z2 = (w1@a1)+b1
            a2 = util.sigmoid(z2)
            logger.debug("Layer 2 Activation : \n{}".format(util.sigmoid(a2)))

            z3 = (w2@a2)+b2
            a3 = util.sigmoid(z3)
            logger.debug("Layer 3 Activation : \n{}".format(a3))

            if i % 100 == random.randint(0,7):
                logger.debug("In : {}\nOut : {}".format(a1.T, a3.T))

            d3 = a1-a3
            # logger.debug("Layer 3 Delta: \n{}".format(d3))

            d2 = util.dsigmoid(z2)*(w2.T@d3)
            # logger.debug("Layer 2 Delta: \n{}".format(d2))

            w2delta = learning_rate * (np.outer(d3, a2))
            w2 += w2delta
            logger.debug("Weights 2: \n{}".format(w2))
            # logger.debug("Weights 2 Delta: \n{}".format(w2delta))

            w1delta = learning_rate * (np.outer(d2, a1))
            w1 += w1delta
            logger.debug("Weights 1: \n{}".format(w1))
            # logger.debug("Weights 1 Delta: \n{}".format(w1delta))

            b2 += learning_rate * d3
            # logger.debug("Layer 2 Bias : \n{}".format(b2))
            # logger.debug("Layer 2 Bias Delta : \n{}".format(learning_rate * d3))

            b1 += learning_rate * d2
            # logger.debug("Layer 1 Bias : \n{}".format(b1))
            # logger.debug("Layer 2 Bias Delta : \n{}".format(learning_rate * d2))

            error_sum += abs(np.sum(np.absolute(d3)))
        errors.append(error_sum)
        if i % 100 == 0: 
          logger.info("Overall error after {} iterations : {}".format(i, error_sum))

    logger.info("Final error after {} iterations : {}".format(iterations, error_sum))
    return errors


if __name__ == '__main__':
    neural_net(1000, 0.7)






