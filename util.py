# http://stackoverflow.com/questions/6534430/why-does-pythons-itertools-permutations-contain-duplicates-when-the-original
def unique(iterable):
    seen = set()
    for x in iterable:
        if x in seen:
            continue
        seen.add(x)
        yield x
