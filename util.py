import numpy as np


# http://stackoverflow.com/questions/6534430/why-does-pythons-itertools-permutations-contain-duplicates-when-the-original
def unique(iterable):
    seen = set()
    for x in iterable:
        if x in seen:
            continue
        seen.add(x)
        yield x


def sigmoid_normal(x):
    import math
    return 1 / (1 + math.exp(-x))


def dsigmoid_normal(x):
    return sigmoid_normal(x) * (1 - sigmoid_normal(x))

sigmoid = np.vectorize(sigmoid_normal, otypes=[np.float])
dsigmoid = np.vectorize(dsigmoid_normal, otypes=[np.float])