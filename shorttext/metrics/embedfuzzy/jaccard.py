
from itertools import product

import numpy as np
from scipy.spatial.distance import cosine

from shorttext.utils import tokenize


def jaccardscore_sents(sent1, sent2, wvmodel, sim_words=lambda vec1, vec2: 1-cosine(vec1, vec2)):
    """ Compute the Jaccard score between sentences based on their word similarities.

    :param sent1: first sentence
    :param sent2: second sentence
    :param wvmodel: word-embeding model
    :param sim_words: function for calculating the similarities between a pair of word vectors (default: cosine)
    :return: soft Jaccard score
    :type sent1: str
    :type sent2: str
    :type wvmodel: gensim.models.keyedvectors.KeyedVectors
    :type sim_words: function
    :rtype: float
    """
    tokens1 = tokenize(sent1)
    tokens2 = tokenize(sent2)
    tokens1 = list(filter(lambda w: w in wvmodel, tokens1))
    tokens2 = list(filter(lambda w: w in wvmodel, tokens2))
    allowable1 = [True] * len(tokens1)
    allowable2 = [True] * len(tokens2)

    simdict = {(i, j): sim_words(wvmodel[tokens1[i]], wvmodel[tokens2[j]])
               for i, j in product(range(len(tokens1)), range(len(tokens2)))}

    intersection = 0.0
    simdictitems = sorted(simdict.items(), key=lambda s: s[1], reverse=True)
    for idxtuple, sim in simdictitems:
        i, j = idxtuple
        if allowable1[i] and allowable2[j]:
            intersection += sim
            allowable1[i] = False
            allowable2[j] = False

    union = len(tokens1) + len(tokens2) - intersection

    if union > 0:
        return intersection / union
    elif intersection == 0:
        return 1.
    else:
        return np.inf
