
from itertools import product
from typing import Optional
import warnings

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix
from scipy.optimize import linprog, OptimizeResult
from gensim.models.keyedvectors import KeyedVectors

from ...utils.gensim_corpora import tokens_to_fracdict


def word_mover_distance_linprog(
        first_sent_tokens: list[str],
        second_sent_tokens: list[str],
        wvmodel: KeyedVectors,
        distancefunc: Optional[callable] = None
) -> OptimizeResult:
    """Compute Word Mover's distance via linear programming.

    Uses scipy.optimize.linprog to compute the transport problem
    for the Word Mover's Distance.

    Args:
        first_sent_tokens: First list of tokens.
        second_sent_tokens: Second list of tokens.
        wvmodel: Word embedding model.
        distancefunc: Distance function for word vectors. Default: Euclidean.

    Returns:
        scipy.optimize.OptimizeResult containing the optimization result.

    Reference:
        Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger,
        "From Word Embeddings to Document Distances," ICML 2015.
    """
    if distancefunc is None:
        distancefunc = euclidean

    nb_tokens_first_sent = len(first_sent_tokens)
    nb_tokens_second_sent = len(second_sent_tokens)

    all_tokens = list(set(first_sent_tokens+second_sent_tokens))
    wordvecs = {token: wvmodel[token].astype(np.float64) for token in all_tokens}

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


def word_mover_distance(
        first_sent_tokens: list[str],
        second_sent_tokens: list[str],
        wvmodel: KeyedVectors,
        distancefunc: Optional[callable] = None
) -> float:
    """Compute Word Mover's distance between token lists.

    Uses word embeddings to compute the minimum transport cost
    between words in two sentences.

    Args:
        first_sent_tokens: First list of tokens.
        second_sent_tokens: Second list of tokens.
        wvmodel: Word embedding model.
        distancefunc: Distance function for word vectors. Default: Euclidean.

    Returns:
        The Word Mover's distance (lower is more similar).

    Reference:
        Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger,
        "From Word Embeddings to Document Distances," ICML 2015.
    """
    if distancefunc is None:
        distancefunc = euclidean

    linprog_result = word_mover_distance_linprog(
        first_sent_tokens,
        second_sent_tokens,
        wvmodel,
        distancefunc=distancefunc
    )

    return linprog_result['fun']
