Metrics
=======

The package `shorttext` provides a few metrics that measure the distances of some kind. They are all
under :module:`shorttext.metrics`. The soft Jaccard score is based on spellings, and the Word Mover's
distance (WMD) embedded word vectors.

Edit Distance and Soft Jaccard Score
------------------------------------

Edit distance, or Damerau-Levenshtein distance, measures the differences
between two words due to insertion, deletion, transposition, substitution etc.
Each of this change causes a distance of 1. The algorithm was written in C.

First import the package:

>>> from shorttext.metrics.dynprog import damerau_levenshtein, longest_common_prefix, similarity, soft_jaccard_score

The distance can be calculated by:

>>> damerau_levenshtein('diver', 'driver')        # insertion, gives 1
>>> damerau_levenshtein('driver', 'diver')        # deletion, gives 1
>>> damerau_levenshtein('topology', 'tooplogy')   # transposition, gives 1
>>> damerau_levenshtein('book', 'blok')           # subsitution, gives 1

The longest common prefix finds the length of common prefix:

>>> longest_common_prefix('topology', 'topological')    # gives 7
>>> longest_common_prefix('police', 'policewoman')      # gives 6

The similarity between words is defined as the larger of the following:

:math:`s = 1 - \frac{\text{DL distance}}{\max( \text(len(word1)), \text(len(word2)) )}`
and
:math:`s = \frac{\text{longest common prefix}}{\max( \text(len(word1)), \text(len(word2)) )}`

>>> similarity('topology', 'topological')    # gives 0.6363636363636364
>>> similarity('book', 'blok')               # gives 0.75

Given the similarity, we say that the intersection, for example, between 'book' and 'blok', has 0.75 elements, or the
union has 1.25 elements. Then the similarity between two sets of tokens can be measured using Jaccard index, with this
"soft" numbers of intersection. Therefore,

>>> soft_jaccard_score(['book', 'seller'], ['blok', 'sellers'])     # gives 0.6716417910447762
>>> soft_jaccard_score(['police', 'station'], ['policeman'])        # gives 0.2857142857142858

The functions `damerau_levenshtein` and `longest_common_prefix` are implemented using Cython_ .
(Before release 0.7.2, they were interfaced to Python using SWIG_ (Simplified Wrapper and Interface Generator)).

Word Mover's Distance
---------------------

Unlike soft Jaccard score that bases similarity on the words' spellings, Word Mover's distance (WMD)
the embedded word vectors. WMD is a special case for Earth Mover's distance (EMD), or Wasserstein
distance. The calculation of WMD in this package is based on linear programming, and the distance between
words are the Euclidean distance by default (not cosine distance), but user can set it accordingly.

Import the modules, and load the word-embedding models:

>>> from shorttext.metrics.wasserstein import word_mover_distance
>>> from shorttext.utils import load_word2vec_model
>>> wvmodel = load_word2vec_model('/path/to/model_file.bin')

Examples:

>>> word_mover_distance(['police', 'station'], ['policeman'], wvmodel)                      # gives 3.060708999633789
>>> word_mover_distance(['physician', 'assistant'], ['doctor', 'assistants'], wvmodel)      # gives 2.276337146759033

More examples can be found in this `IPython Notebook
<https://github.com/stephenhky/PyWMD/blob/master/WordMoverDistanceDemo.ipynb>`_ .

In `gensim`, the Word2Vec model allows the calculation of WMD if user installed the package PyEMD_. It is based on the
scale invariant feature transform (SIFT), an algorithm for EMD based on L1-distance (Manhattan distance).
For more details,
please refer to their `tutorial
<https://radimrehurek.com/gensim/models/keyedvectors.html>`_ , and cite the two papers by Ofir Pele and Michael Werman
if it is used.

Jaccard Index Due to Cosine Distances
-------------------------------------

In the above section of edit distance, the Jaccard score was calculated by considering soft membership
using spelling. However, we can also compute the soft membership by cosine similarity with

>>> from shorttext.utils import load_word2vec_model
>>> wvmodel = load_word2vec_model('/path/to/model_file.bin')
>>> from shorttext.metrics.embedfuzzy import jaccardscore_sents

For example, the number of words between the set containing 'doctor' and that containing 'physician'
is 0.78060223420956831 (according to Google model), and therefore the Jaccard score is

:math:`0.78060223420956831 / (2-0.78060223420956831) = 0.6401538990056869`

And it can be seen by running it:

>>> jaccardscore_sents('doctor', 'physician', wvmodel)   # gives 0.6401538990056869
>>> jaccardscore_sents('chief executive', 'computer cluster', wvmodel)   # gives 0.0022515450768836143
>>> jaccardscore_sents('topological data', 'data of topology', wvmodel)   # gives 0.67588977344632573

Reference
---------

"Damerau-Levenshtein Distance." [`Wikipedia
<https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance>`_]

"Jaccard index." [`Wikipedia
<https://en.wikipedia.org/wiki/Jaccard_index>`_]

Daniel E. Russ, Kwan-Yuet Ho, Calvin A. Johnson, Melissa C. Friesen, "Computer-Based Coding of Occupation Codes for Epidemiological Analyses," *2014 IEEE 27th International Symposium on Computer-Based Medical Systems* (CBMS), pp. 347-350. (2014) [`IEEE
<http://ieeexplore.ieee.org/abstract/document/6881904/>`_]

Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger, "From Word Embeddings to Document Distances," *ICML* (2015).

Ofir Pele, Michael Werman, "A linear time histogram metric for improved SIFT matching," *Computer Vision - ECCV 2008*, 495-508 (2008). [`ACM
<http://dl.acm.org/citation.cfm?id=1478212>`_]

Ofir Pele, Michael Werman, "Fast and robust earth mover's distances," *Proc. 2009 IEEE 12th Int. Conf. on Computer Vision*, 460-467 (2009). [`IEEE
<http://ieeexplore.ieee.org/document/5459199/>`_]

"Word Mover’s Distance as a Linear Programming Problem," *Everything About Data Analytics*, WordPress (2017). [`WordPress
<https://datawarrior.wordpress.com/2017/08/16/word-movers-distance-as-a-linear-programming-problem/>`_]


Home: :doc:`index`

.. _SWIG: http://www.swig.org/
.. _PyEMD: https://github.com/wmayner/pyemd
.. _Cython: http://cython.org/