
import numpy as np
import numba as nb


@nb.njit
def longest_common_prefix(word1: str, word2: str) -> int:
    lcp = 0
    for i in range(min(len(word1), len(word2))):
        if word1[i] == word2[i]:
            lcp += 1
        else:
            break
    return lcp
