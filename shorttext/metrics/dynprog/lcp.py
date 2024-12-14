
import numba as nb


@nb.njit
def longest_common_prefix(word1: str, word2: str) -> int:
    """ Calculate the longest common prefix (LCP) between two words.

    :param word1: first word
    :param word2: seccond word
    :return: longest common prefix (LCP)
    :type word1: str
    :type word2: str
    :rtype: int
    """
    lcp = 0
    for i in range(min(len(word1), len(word2))):
        if word1[i] == word2[i]:
            lcp += 1
        else:
            break
    return lcp
