Introduction
============

This package `shorttext` is a Python package that facilitates supervised
learning for short text categorization. Due to the sparseness of words and
the lack of information carried in the short texts themselves, an intermediate
representation of the texts and documents are needed before they are put into
any classification algorithm. In this package, it facilitates various types
of these representations, including topic modeling and word-embedding algorithms.

Characteristics:

- example data provided (including subject keywords and NIH RePORT); (see :doc:`tutorial_dataprep`)
- text preprocessing; (see :doc:`tutorial_textpreprocessing`)
- pre-trained word-embedding support; (see :doc:`tutorial_wordembed`)
- `gensim` topic models (LDA, LSI, Random Projections) and autoencoder; (see :doc:`tutorial_topic`)
- topic model representation supported for supervised learning using `scikit-learn`; (see :doc:`tutorial_topic`)
- cosine distance classification; (see :doc:`tutorial_topic`, :doc:`tutorial_umvec`)
- neural network classification (including ConvNet, and C-LSTM); (see :doc:`tutorial_nnlib`) and
- maximum entropy classification. (see :doc:`tutorial_maxent`)

Author: Kwan-Yuet Ho (LinkedIn_, ResearchGate_, Twitter_)

Home: :doc:`index`

.. _LinkedIn: https://www.linkedin.com/in/kwan-yuet-ho-19882530
.. _ResearchGate: https://www.researchgate.net/profile/Kwan-yuet_Ho
.. _Twitter: https://twitter.com/stephenhky