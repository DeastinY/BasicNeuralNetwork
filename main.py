import util
import numpy as np
import logging
import itertools

if __name__ == "__main__":
    np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})
    logging.basicConfig(level=logging.DEBUG)
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

    iterations = 100
    learning_rate = 1

    for i in range(iterations):
        for j in learning_examples:
            a1 = np.array([j]).T
            logger.debug("Layer 1 Activation : \n{}".format(a1))

            a2 = util.sigmoid((w1@a1)+b1)
            logger.debug("Layer 2 Activation : \n{}".format(a2))

            a3 = util.sigmoid((w2@a2)+b2)
            logger.debug("Layer 3 Activation : \n{}".format(a3))

            d3 = util.dsigmoid(a3)*(a3-a1)
            logger.debug("Layer 3 Delta: \n{}".format(d3))

            d2 = util.dsigmoid(a2)*(w2.T@d3)
            logger.debug("Layer 2 Delta: \n{}".format(d2))

            d1 = util.dsigmoid(a1)*(w1.T@d2)
            logger.debug("Layer 1 Delta: \n{}".format(d3))

            # self.weights[i] = self.weights[i]-self.learning_rate*np.outer(self.errors[i+1],self.outputs[i])
            print(w2)

            w2 -= learning_rate * np.outer(d3, a3)
            logger.debug("Weights 2: \n{}".format(w2))

            w1 += (d2 @ a1.T) / iterations
            logger.debug("Delta Weights 1: \n{}".format(w1))
            w1 += learning_rate * w2
            bw1 = np.c_[np.full((3, 1), 1/iterations, dtype=int), w1]

    logger.info("Output after {} iterations : \n{}".format(iterations, d3))









