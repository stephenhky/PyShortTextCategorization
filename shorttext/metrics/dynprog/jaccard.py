
from itertools import product

from .dldist import damerau_levenshtein
from .lcp import longest_common_prefix


def similarity(word1, word2):
    """ Return the similarity between the two words.

    Return the similarity between the two words, between 0 and 1 inclusively.
    The similarity is the maximum of the two values:
    - 1 - Damerau-Levenshtein distance between two words / maximum length of the two words
    - longest common prefix of the two words / maximum length of the two words

    Reference: Daniel E. Russ, Kwan-Yuet Ho, Calvin A. Johnson, Melissa C. Friesen, "Computer-Based Coding of Occupation Codes for Epidemiological Analyses," *2014 IEEE 27th International Symposium on Computer-Based Medical Systems* (CBMS), pp. 347-350. (2014) [`IEEE
    <http://ieeexplore.ieee.org/abstract/document/6881904/>`_]

    :param word1: a word
    :param word2: a word
    :return: similarity, between 0 and 1 inclusively
    :type word1: str
    :type word2: str
    :rtype: float
    """
    maxlen = max(len(word1), len(word2))
    editdistance = damerau_levenshtein(word1, word2)
    lcp = longest_common_prefix(word1, word2)
    return max(1. - float(editdistance)/maxlen, float(lcp)/maxlen)


def soft_intersection_list(tokens1, tokens2):
    """ Return the soft number of intersections between two lists of tokens.

    :param tokens1: list of tokens.
    :param tokens2: list of tokens.
    :return: soft number of intersections.
    :type tokens1: list
    :type tokens2: list
    :rtype: float
    """
    intersected_list = [((token1, token2), similarity(token1, token2)) for token1, token2 in product(tokens1, tokens2)]
    intersected_list = sorted(intersected_list, key=lambda item: item[1], reverse=True)

    included_list = set()
    used_tokens1 = set()
    used_tokens2 = set()
    for (token1, token2), sim in intersected_list:
        if (not (token1 in used_tokens1)) and (not (token2 in used_tokens2)):
            included_list.add(((token1, token2), sim))
            used_tokens1.add(token1)
            used_tokens2.add(token2)

    return included_list


def soft_jaccard_score(tokens1, tokens2):
    """ Return the soft Jaccard score of the two lists of tokens, between 0 and 1 inclusively.

    Reference: Daniel E. Russ, Kwan-Yuet Ho, Calvin A. Johnson, Melissa C. Friesen, "Computer-Based Coding of Occupation Codes for Epidemiological Analyses," *2014 IEEE 27th International Symposium on Computer-Based Medical Systems* (CBMS), pp. 347-350. (2014) [`IEEE
    <http://ieeexplore.ieee.org/abstract/document/6881904/>`_]

    :param tokens1: list of tokens.
    :param tokens2: list of tokens.
    :return: soft Jaccard score, between 0 and 1 inclusively.
    :type tokens1: list
    :type tokens2: list
    :rtype: float
    """
    intersection_list = soft_intersection_list(tokens1, tokens2)
    num_intersections = sum([item[1] for item in intersection_list])
    num_unions = len(tokens1) + len(tokens2) - num_intersections
    return float(num_intersections)/float(num_unions)
