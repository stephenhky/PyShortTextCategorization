
from itertools import product

from .dldist import damerau_levenshtein

def similarity(word1, word2):
    maxlen = max(len(word1), len(word2))
    editdistance = damerau_levenshtein(word1, word2)
    return 1. - float(editdistance)/maxlen

def soft_intersection_list(tokens1, tokens2):
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
    intersection_list = soft_intersection_list(tokens1, tokens2)
    num_intersections = sum(map(lambda item: item[1], intersection_list))
    num_unions = len(tokens1) + len(tokens2) - num_intersections
    return float(num_intersections)/float(num_unions)
