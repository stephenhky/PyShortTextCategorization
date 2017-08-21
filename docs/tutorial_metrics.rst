Metrics
=======

The package `shorttext` provides a few metrics that measure the distances of some kind. They are all
under :module:`shorttext.metrics`.

Edit Distance and Soft Jaccard Score
------------------------------------

Edit distance, or Damerau-Levenshtein distance, measures the differences
between two words due to insertion, deletion, transposition, substitution etc.
Each of this change causes a distance of 1. The algorithm was written in C.

FIrst import the package:

>>> from shorttext.metrics.dynprog import damerau_levenshtein, longest_common_prefix, jaccard

The distance can be calculated by:

>>> damerau_levenshtein('diver', 'driver')        # insertion, gives 1
>>> damerau_levenshtein('driver', 'diver')        # deletion, gives 1
>>> damerau_levenshtein('topology', 'tooplogy')   # transposition, gives 1
>>> damerau_levenshtein('book', 'blok')           # subsitution, gives 1

The longest common prefix finds the length of common prefix:

>>> longest_common_prefix('topology', 'topological')    # gives 7
>>> longest_common_prefix('police', 'policewoman')      # gives 6



Word Mover's Distance
---------------------




Reference
---------

"Damerau-Levenshtein Distance." [`Wikipedia
<https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance>`_]

"Jaccard index." [`Wikipedia
<https://en.wikipedia.org/wiki/Jaccard_index>`_]

Daniel E. Russ, Kwan-Yuet Ho, Calvin A. Johnson, Melissa C. Friesen, "Computer-Based Coding of Occupation Codes for Epidemiological Analyses," *2014 IEEE 27th International Symposium on Computer-Based Medical Systems* (CBMS), pp. 347-350. (2014) [`IEEE
<http://ieeexplore.ieee.org/abstract/document/6881904/>`_]

Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger, "From Word Embeddings to Document Distances," *ICML* (2015).
