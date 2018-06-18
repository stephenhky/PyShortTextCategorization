
import numpy as np
cimport numpy as np

def longest_common_prefix(str word1, str word2):
    cdef int len1 = len(word1)
    cdef int len2 = len(word2)

    cdef int lcp = 0
    cdef int i

    for i in range(min(len1, len2)):
        if word1[i] == word2[i]:
            lcp += 1
        else:
            break

    return lcp

