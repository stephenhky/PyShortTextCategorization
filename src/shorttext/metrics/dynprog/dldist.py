
import numpy as np
import numba as nb


@nb.njit
def damerau_levenshtein(word1: str, word2: str) -> int:
    """ Calculate the Demarau-Levenshtein (DL) distance between two words.

    :param word1: first word
    :param word2: seccond word
    :return: Damerau-Levenshtein (DL) distance
    :type word1: str
    :type word2: str
    :rtype: int
    """
    len1 = len(word1)
    len2 = len(word2)
    matrix = np.zeros((len1+1, len2+1), dtype=np.int8)

    for i in range(len1+1):
        matrix[i, 0] = i
    for j in range(len2+1):
        matrix[0, j] = j

    for i in range(len1+1):
        for j in range(len2+1):
            cost = 0
            if i > 0 and j > 0 and (word1[i-1] != word2[j-1]):
                cost = 1
            delcost = matrix[i-1, j] + 1
            inscost = matrix[i, j-1] + 1
            subcost = matrix[i-1, j-1] + cost
            score = min(min(delcost, inscost), subcost)
            if ((i > 1) & (j > 1) & (word1[i - 1] == word2[j - 2]) & (word1[i - 2] == word2[j - 1])):
                score = min(score, matrix[i-2, j-2] + cost)
            matrix[i, j] = score

    print(matrix)

    return matrix[len1, len2]
