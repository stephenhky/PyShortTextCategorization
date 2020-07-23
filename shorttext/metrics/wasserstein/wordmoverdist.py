
from itertools import product
import warnings

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix
from scipy.optimize import linprog

from shorttext.utils.gensim_corpora import tokens_to_fracdict


def word_mover_distance_linprog(first_sent_tokens, second_sent_tokens, wvmodel, distancefunc=euclidean):
    """ Compute the Word Mover's distance (WMD) between the two given lists of tokens, and return the LP problem class.

    Using methods of linear programming, supported by PuLP, calculate the WMD between two lists of words. A word-embedding
    model has to be provided. The whole `scipy.optimize.Optimize` object is returned.

    Reference: Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger, "From Word Embeddings to Document Distances," *ICML* (2015).

    :param first_sent_tokens: first list of tokens.
    :param second_sent_tokens: second list of tokens.
    :param wvmodel: word-embedding models.
    :param distancefunc: distance function that takes two numpy ndarray.
    :return: the whole result of the linear programming problem
    :type first_sent_tokens: list
    :type second_sent_tokens: list
    :type wvmodel: gensim.models.keyedvectors.KeyedVectors
    :type distancefunc: function
    :rtype: scipy.optimize.OptimizeResult
    """
    nb_tokens_first_sent = len(first_sent_tokens)
    nb_tokens_second_sent = len(second_sent_tokens)

    all_tokens = list(set(first_sent_tokens+second_sent_tokens))
    wordvecs = {token: wvmodel[token] for token in all_tokens}

    first_sent_buckets = tokens_to_fracdict(first_sent_tokens)
    second_sent_buckets = tokens_to_fracdict(second_sent_tokens)

    collapsed_idx_func = lambda i, j: i*nb_tokens_second_sent + j

    # assigning T
    T = np.zeros(nb_tokens_first_sent*nb_tokens_second_sent)
    for i, j in product(range(nb_tokens_first_sent), range(nb_tokens_second_sent)):
        T[collapsed_idx_func(i, j)] = distancefunc(wordvecs[first_sent_tokens[i]],
                                                   wordvecs[second_sent_tokens[j]])

    # assigning Aeq and beq
    Aeq = csr_matrix(
        (nb_tokens_first_sent+nb_tokens_second_sent,
         nb_tokens_first_sent*nb_tokens_second_sent)
    )
    beq = np.zeros(nb_tokens_first_sent+nb_tokens_second_sent)
    for i in range(nb_tokens_first_sent):
        for j in range(nb_tokens_second_sent):
            Aeq[i, collapsed_idx_func(i, j)] = 1.
        beq[i] = first_sent_buckets[first_sent_tokens[i]]
    for j in range(nb_tokens_second_sent):
        for i in range(nb_tokens_first_sent):
            Aeq[j+nb_tokens_first_sent, collapsed_idx_func(i, j)] = 1.
        beq[j+nb_tokens_first_sent] = second_sent_buckets[second_sent_tokens[j]]

    return linprog(T, A_eq=Aeq, b_eq=beq)


def word_mover_distance(first_sent_tokens, second_sent_tokens, wvmodel, distancefunc=euclidean, lpFile=None):
    """ Compute the Word Mover's distance (WMD) between the two given lists of tokens.

    Using methods of linear programming, calculate the WMD between two lists of words. A word-embedding
    model has to be provided. WMD is returned.

    Reference: Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger, "From Word Embeddings to Document Distances," *ICML* (2015).

    :param first_sent_tokens: first list of tokens.
    :param second_sent_tokens: second list of tokens.
    :param wvmodel: word-embedding models.
    :param distancefunc: distance function that takes two numpy ndarray.
    :param lpFile: deprecated, kept for backward incompatibility. (default: None)
    :return: Word Mover's distance (WMD)
    :type first_sent_tokens: list
    :type second_sent_tokens: list
    :type wvmodel: gensim.models.keyedvectors.KeyedVectors
    :type distancefunc: function
    :type lpFile: str
    :rtype: float
    """
    linprog_result = word_mover_distance_linprog(first_sent_tokens, second_sent_tokens, wvmodel,
                                                 distancefunc=distancefunc)
    if lpFile is not None:
        warnings.warn('The parameter `lpFile` (value: {}) is not used; parameter is deprecated as ' + \
                      'the package `pulp` is no longer used. Check your code if there is a dependency on ' + \
                      'this parameter.')
    return linprog_result['fun']
