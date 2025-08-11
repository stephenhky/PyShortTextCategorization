Introduction
============

This package `shorttext` is a Python package that facilitates supervised and unsupervised
learning for short text categorization. Due to the sparseness of words and
the lack of information carried in the short texts themselves, an intermediate
representation of the texts and documents are needed before they are put into
any classification algorithm. In this package, it facilitates various types
of these representations, including topic modeling and word-embedding algorithms.

The package `shorttext` runs on Python 3.9, 3.10, 3.11, and 3.12.

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
- character-level sequence-to-sequence (seq2seq) learning; (see :doc:`tutorial_charbaseseq2seq`)
- spell correction; (see :doc:`tutorial_spell`)

Author: Kwan Yuet Stephen Ho (LinkedIn_, ResearchGate_)
Other contributors: `Chinmaya Pancholi <https://www.linkedin.com/in/cpancholi>`_, `Minseo Kim <https://kmingseo.github.io/>`_

Home: :doc:`index`

.. _LinkedIn: https://www.linkedin.com/in/kwan-yuet-ho-19882530
.. _ResearchGate: https://www.researchgate.net/profile/Kwan-yuet_Ho
