Word-Embedding Cosine Similarity Classifier
===========================================

Given a pre-trained word-embedding models like Word2Vec, a classifier
based on cosine similarities can be built. In training the data,
the embedded vectors in every word in that class are averaged. The
score for a given text to each class is the cosine similarity between the averaged
vector of the given text and the precalculated vector of that class.

See: :doc:`tutorial_wordembed` .

To load the Word2Vec model,

>>> from shorttext.utils.wordembed import load_word2vec_model
>>> wvmodel = load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

