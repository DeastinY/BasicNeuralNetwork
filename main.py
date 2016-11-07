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

    iterations = 1
    learning_rate = 1

    for i in range(iterations):
        for j in learning_examples:
            a1 = np.array([j]).T
            logger.debug("Layer 1 Activation : \n{}".format(util.sigmoid(a1)))

            a2 = (w1@a1)+b1
            logger.debug("Layer 2 Activation : \n{}".format(util.sigmoid(a2)))

            a3 = (w2@a2)+b2
            logger.debug("Layer 3 Activation : \n{}".format(util.sigmoid(a3)))

            # self.errors[i] = self.fprimes[i](self.inputs[i])*self.weights[i].T.dot(self.errors[i+1])

            d3 = util.sigmoid(a3) - a1
            logger.debug("Layer 3 Delta: \n{}".format(d3))

            d2 = util.dsigmoid(a2)*(w2.T@d3)
            logger.debug("Layer 2 Delta: \n{}".format(d2))

            # self.weights[i] = self.weights[i]-self.learning_rate*np.outer(self.errors[i+1],self.outputs[i])

            w2 -= learning_rate * (np.outer(d3, util.sigmoid(a2)))
            logger.debug("Weights 2: \n{}".format(w2))

            w1 -= learning_rate * (np.outer(d2, util.sigmoid(a1)))
            logger.debug("Weights 1: \n{}".format(w1))

            # self.biases[i] = self.biases[i] - self.learning_rate*self.errors[i+1]

            b2 -= learning_rate * d3
            logger.debug("Layer 2 Bias : \n{}".format(b2))

            b1 -= learning_rate * d2
            logger.debug("Layer 1 Bias : \n{}".format(b1))

    logger.info("Output after {} iterations : \n{}".format(iterations, d3))









