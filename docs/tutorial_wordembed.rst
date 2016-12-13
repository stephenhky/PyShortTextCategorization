Word Embedding Models
=====================

Word2Vec
--------

The most commonly used word-embedding model is Word2Vec. Its model can be downloaded from
their page. To load the model, call:

>>> import shorttext
>>> wvmodel = shorttext.utils.wordembed.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be `binary=True`. In fact, it is equivalent to calling:

>>> from gensim.models import Word2Vec
>>> wvmodel =Word2Vec.load_word2vec_format('/path/to/GoogleNews-vectors-negative300.bin.gz', binary=True)

Word2Vec is a neural network model that embeds words into semantic vectors that carry semantic meaning.


Links
-----

- Word2Vec_

Reference
---------

Jeffrey Pennington, Richard Socher, Christopher D. Manning, “GloVe: Global Vectors for Word Representation,” *Empirical Methods in Natural Language Processing (EMNLP)*, pp. 1532-1543 (2014). [`PDF
<http://www.aclweb.org/anthology/D14-1162>`_]

Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean, “Efficient Estimation of Word Representations in Vector Space,” *ICLR* 2013 (2013). [`arXiv
<https://arxiv.org/abs/1301.3781>`_]

.. _Word2Vec: https://code.google.com/archive/p/word2vec/
