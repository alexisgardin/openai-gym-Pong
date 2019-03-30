"""
Nothing for the moment, juste load data
"""

from os import listdir
from os.path import isfile, join
import pickle
import numpy as np


def load():
    """
    load data from run
    :return: an array of 2-tuple,
    the first element represent the image
    with a matrix of 160x160 and the second
    list is the action
    """
    mypath = 'save/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    loadData = []
    for f in onlyfiles:
        with open(mypath + f, 'rb') as pick:
            # The protocol version used is detected automatically, so we do not
            # have to specify it.
            loadData = loadData + pickle.load(pick)
    loadData = np.array(loadData)
    return loadData


if __name__ == '__main__':
    data = load()
