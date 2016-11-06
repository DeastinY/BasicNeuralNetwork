import math
import util
import numpy as np
import logging
import itertools


def g(z):
    return 1/(1+math.exp(-z))

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info("Program started")

    learning_examples = [l for l in util.unique(itertools.permutations('00000001'))]
    learning_examples = np.array(learning_examples, dtype=int)
    logger.debug("Learning Examples : \n{}".format(learning_examples))

    for i in learning_examples.T:
        i = i.reshape(-1, 1)
        # Add Bias
        bi = np.r_[[[1]], i]
        logger.debug("Learning Examples plus Bias: \n{}".format(bi))

        # Forward Propagation

        # Hidden Layer
        logger.info("Forward Propagation : Hidden Layer")
        hidden_weights = np.c_[np.full((3, 1), 1, dtype=int), np.full((3, 8), 0, dtype=int)]
        logger.debug("Hidden Weights plus Bias : \n{}".format(hidden_weights))
        hidden_activations = hidden_weights @ bi
        np.apply_along_axis(g, 1, hidden_activations)
        logger.debug("Hidden Activations : \n{}".format(hidden_activations))

        # Add Bias
        bhidden_activations = np.r_[[[1]], hidden_activations]
        logger.debug("Hidden Activations plus Bias : \n{}".format(bhidden_activations))

        # Output Layer
        logger.info("Forward Propagation : Output Layer")
        output_weights = np.c_[np.full((8, 1), 1, dtype=int), np.full((8, 3), 0, dtype=int)]
        logger.debug("Output Weights plus Bias : \n{}".format(output_weights))
        output_activations = output_weights @ bhidden_activations
        np.apply_along_axis(g, 1, output_activations)
        logger.debug("Output Activations : \n{}".format(output_activations))

        # Back Propagation

        error_output = output_activations - i
        logger.debug("Error Output : \n{}".format(error_output))

        logger.info("Backpropagation : Output Layer")
        error_hidden_activation = hidden_activations * (1 - hidden_activations)
        error_hidden = np.transpose(np.delete(hidden_weights, 0, 1) @ error_output) * error_hidden_activation
        logger.debug("Error Hidden : \n{}".format(error_hidden))

        # Update Weights
        hidden_weights = hidden_weights + (hidden_activations @ error_output)

        input()








