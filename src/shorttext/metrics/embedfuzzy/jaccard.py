
from itertools import product
from typing import Optional

import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from ...utils import tokenize
from ...utils.compute import cosine_similarity


def jaccardscore_sents(
        sent1: str,
        sent2: str,
        wvmodel: KeyedVectors,
        sim_words: Optional[callable] = None
) -> float:
    """Compute Jaccard score between sentences using embeddings.

    Uses word embeddings to compute a fuzzy Jaccard score where
    word similarity is measured via embedding cosine similarity.

    Args:
        sent1: First sentence.
        sent2: Second sentence.
        wvmodel: Word embedding model.
        sim_words: Similarity function for word vectors. Default: cosine.

    Returns:
        Fuzzy Jaccard score between 0 and 1.
    """
    if sim_words is None:
        sim_words = cosine_similarity

    tokens1 = tokenize(sent1)
    tokens2 = tokenize(sent2)
    tokens1 = list(filter(lambda w: w in wvmodel, tokens1))
    tokens2 = list(filter(lambda w: w in wvmodel, tokens2))
    allowable1 = [True] * len(tokens1)
    allowable2 = [True] * len(tokens2)

    simdict = {(i, j): sim_words(wvmodel[tokens1[i]].astype(np.float64), wvmodel[tokens2[j]].astype(np.float64))
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
