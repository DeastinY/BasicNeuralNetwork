import util
import logging
import itertools
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info("Program started")

    learning_examples = list(util.unique(itertools.permutations('00000001')))
    logger.debug("Learning Examples : {}".format([ "".join(l) for l in learning_examples]))



