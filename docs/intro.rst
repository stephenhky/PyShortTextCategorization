Introduction
============

This package `shorttext` is a Python package that facilitates the supervised
learning for short text categorization. Due to the sparseness of words and
the lack of information carried in the short texts themselves, an intermediate
representation of the texts and documents are needed before they are put into
any classification algorithms. In this package, it facilitates various types
of these representations, including topic modeling and word-embedding algorithms.

Characteristics:

- example data provided (including subject keywords and NIH RePORT); (see :doc:`tutorial_dataprep`)
- text preprocessing; (see :doc:`tutorial_textpreprocessing`)
- pre-trained word-embedding support; (see :doc:`tutorial_wordembed`)
- `gensim` topic models (LDA, LSI, Random Projections) and autoencoder; (see :doc:`tutorial_topic`)
- topic model representation supported for supervised learning using `scikit-learn`; (see :doc:`tutorial_topic`)
- cosine distance classification; (see :doc:`tutorial_topic`, :doc:`tutorial_umvec`) and
- neural network classification (including ConvNet, and C-LSTM). (see :doc:`tutorial_nnlib`)