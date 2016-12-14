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
>>> wvmodel = Word2Vec.load_word2vec_format('/path/to/GoogleNews-vectors-negative300.bin.gz', binary=True)

Word2Vec is a neural network model that embeds words into semantic vectors that carry semantic meaning.

GloVe
-----

Stanford NLP Group developed a similar word-embedding algorithm, with a good theory explaining how
it works. It is extremely similar to Word2Vec.

One can convert a text-format GloVe model into a text-format Word2Vec model. More information can be found
in the documentation of `gensim`: `Converting GloVe to Word2Vec
<https://radimrehurek.com/gensim/scripts/glove2word2vec.html>`_

Links
-----

- Word2Vec_
- GloVe_

Reference
---------

Jeffrey Pennington, Richard Socher, Christopher D. Manning, “GloVe: Global Vectors for Word Representation,” *Empirical Methods in Natural Language Processing (EMNLP)*, pp. 1532-1543 (2014). [`PDF
<http://www.aclweb.org/anthology/D14-1162>`_]

Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean, “Efficient Estimation of Word Representations in Vector Space,” *ICLR* 2013 (2013). [`arXiv
<https://arxiv.org/abs/1301.3781>`_]

Blog Entries
------------

Radim Řehůřek, "Making sense of word2vec," RaRe Technologies (2014). [`RaRe
<https://rare-technologies.com/making-sense-of-word2vec/>`_]

"Probabilistic Theory of Word Embeddings: GloVe," *Everything About Data Analytics*, WordPress (2016). [`WordPress
<https://datawarrior.wordpress.com/2016/07/25/probabilistic-theory-of-word-embeddings-glove/>`_]

"Toying with Word2Vec," *Everything About Data Analytics*, WordPress (2015). [`WordPress
<https://datawarrior.wordpress.com/2015/10/25/codienerd-2-toying-with-word2vec/>`_]


.. _Word2Vec: https://code.google.com/archive/p/word2vec/
.. _GloVe: http://nlp.stanford.edu/projects/glove/