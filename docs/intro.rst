Introduction
============

This package `shorttext` is a Python package that facilitates supervised and unsupervised
learning for short text categorization. Due to the sparseness of words and
the lack of information carried in the short texts themselves, an intermediate
representation of the texts and documents are needed before they are put into
any classification algorithm. In this package, it facilitates various types
of these representations, including topic modeling and word-embedding algorithms.

Since release 1.0.0, `shorttext` runs on Python 2.7, 3.5, and 3.6. Since release 1.0.7,
it runs also in Python 3.7.

Characteristics:

- example data provided (including subject keywords and NIH RePORT); (see :doc:`tutorial_dataprep`)
- text preprocessing; (see :doc:`tutorial_textpreprocessing`)
- pre-trained word-embedding support; (see :doc:`tutorial_wordembed`)
- `gensim` topic models (LDA, LSI, Random Projections) and autoencoder; (see :doc:`tutorial_topic`)
- topic model representation supported for supervised learning using `scikit-learn`; (see :doc:`tutorial_topic`)
- cosine distance classification; (see :doc:`tutorial_topic`, :doc:`tutorial_sumvec`)
- neural network classification (including ConvNet, and C-LSTM); (see :doc:`tutorial_nnlib`)
- maximum entropy classification; (see :doc:`tutorial_maxent`)
- metrics of phrases differences, including soft Jaccard score (using Damerau-Levenshtein distance), and Word Mover's distance (WMD); (see :doc:`tutorial_metrics`)
- character-level sequence-to-sequence (seq2seq) learning; (see :doc:`tutorial_charbaseseq2seq`) and
- spell correction. (see :doc:`tutorial_spell`)

Before release 0.7.2, part of the package was implemented using C, and it is interfaced to
Python using SWIG_ (Simplified Wrapper and Interface Generator). Since 1.0.0, these implementations
were replaced with Cython_.

Author: Kwan-Yuet Ho (LinkedIn_, ResearchGate_, Twitter_)

Home: :doc:`index`

.. _LinkedIn: https://www.linkedin.com/in/kwan-yuet-ho-19882530
.. _ResearchGate: https://www.researchgate.net/profile/Kwan-yuet_Ho
.. _Twitter: https://twitter.com/stephenhky
.. _SWIG: http://www.swig.org/
.. _Cython: http://cython.org/
