import numpy as np
from itertools import chain


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def flip_web(lines):
    x, y = zip(*list(chain.from_iterable(lines)))
    y = np.asarray(y)
    miny = min(y)
    maxy = max(y)

    # Normalise y by setting min to 0
    normy = y - miny
    # Find line of reflection
    normmid = (maxy - miny) / 2
    # Find diff of points from line of reflection
    normdelta = normy - normmid
    # Flip these diffs to invert points
    normflippeddelta = normdelta * -1
    # Apply diffs back to line of reflection to get new normalised points
    normflippedy = normflippeddelta + normmid
    # Revert y normalisation
    flippedy = normflippedy + miny

    # Pack into new list
    newpoints = list(zip(x, flippedy.astype(int).tolist()))
    # Chunk and return
    return [x for x in chunks(newpoints, 2)]