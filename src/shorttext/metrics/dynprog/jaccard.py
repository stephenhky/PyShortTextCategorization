
from itertools import product

from .dldist import damerau_levenshtein
from .lcp import longest_common_prefix


def similarity(word1: str, word2: str) -> float:
    """Calculate similarity between two words.

    Computes similarity as the maximum of:
    - 1 - Damerau-Levenshtein distance / max length
    - Longest common prefix length / max length

    Args:
        word1: First word.
        word2: Second word.

    Returns:
        Similarity score between 0 and 1.

    Reference:
        Daniel E. Russ, Kwan-Yuet Ho, Calvin A. Johnson, Melissa C. Friesen,
        "Computer-Based Coding of Occupation Codes for Epidemiological Analyses,"
        IEEE CBMS 2014, pp. 347-350.
        http://ieeexplore.ieee.org/abstract/document/6881904/
    """
    maxlen = max(len(word1), len(word2))
    editdistance = damerau_levenshtein(word1, word2)
    lcp = longest_common_prefix(word1, word2)
    return max(1. - float(editdistance)/maxlen, float(lcp)/maxlen)


def soft_intersection_list(tokens1: list[str], tokens2: list[str]) -> set[str]:
    """Compute soft intersection between two token lists.

    Finds the best matching pairs between tokens using similarity,
    where each token can only match once.

    Args:
        tokens1: First list of tokens.
        tokens2: Second list of tokens.

    Returns:
        Set of ((token1, token2), similarity) tuples representing matches.
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


def soft_jaccard_score(tokens1: str, tokens2: str) -> float:
    """Compute soft Jaccard score between token lists.

    Uses fuzzy matching based on edit distance and longest common prefix.

    Args:
        tokens1: First list of tokens.
        tokens2: Second list of tokens.

    Returns:
        Soft Jaccard score between 0 and 1.

    Reference:
        Daniel E. Russ, Kwan-Yuet Ho, Calvin A. Johnson, Melissa C. Friesen,
        "Computer-Based Coding of Occupation Codes for Epidemiological Analyses,"
        IEEE CBMS 2014, pp. 347-350.
        http://ieeexplore.ieee.org/abstract/document/6881904/
    """
    intersection_list = soft_intersection_list(tokens1, tokens2)
    num_intersections = sum([item[1] for item in intersection_list])
    num_unions = len(tokens1) + len(tokens2) - num_intersections
    return num_intersections / num_unions
