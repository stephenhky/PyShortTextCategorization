Character-Based Sequence-to-Sequence (seq2seq) Models
=====================================================

Since release 0.6.0, `shorttext` supports sequence-to-sequence (seq2seq) learning. While there is a general seq2seq class
behind, it provides a character-based seq2seq implementation.

Creating One-hot Vectors
------------------------

To use it, create an instance of the class :class:`shorttext.generators.SentenceToCharVecEncoder`:

>>> import numpy as np
>>> import shorttext
>>> from urllib2 import urlopen   # for python 2.7; from urllib.request import urlopen for python 3
>>> chartovec_encoder = shorttext.generators.initSentenceToCharVecEncoder(urlopen('http://norvig.com/big.txt', 'r'))

The above code is the same as :doc:`tutorial_charbaseonehot` .

Training
--------

Then we can train the model by creating an instance of :class:`shorttext.generators.CharBasedSeq2SeqGenerator`:

>>> seq2seqer = shorttext.generators.CharBasedSeq2SeqGenerator(chartovec_encoder, latent_dim, 120)

And then train this neural network model:

>>> seq2seqer.train(text, epochs=100)

This model takes several hours to train on a laptop.

Decoding
--------

After training, we can use this class as a generative model
of answering questions as a chatbot:

>>> seq2seqer.decode('Happy Holiday!')

It does not give definite answers because there is a stochasticity in the prediction.

Model I/O
---------

This model can be saved by entering:

>>> seq2seqer.save_compact_model('/path/to/norvigtxt_iter5model.bin')

And can be loaded by:

>>> seq2seqer2 = shorttext.generators.seq2seq.charbaseS2S.loadCharBasedSeq2SeqGenerator('/path/to/norvigtxt_iter5model.bin')

Reference
---------

Aurelien Geron, *Hands-On Machine Learning with Scikit-Learn and TensorFlow* (Sebastopol, CA: O'Reilly Media, 2017). [`O\'Reilly
<http://shop.oreilly.com/product/0636920052289.do>`_]

Ilya Sutskever, James Martens, Geoffrey Hinton, "Generating Text with Recurrent Neural Networks," *ICML* (2011). [`UToronto
<http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf>`_]

Ilya Sutskever, Oriol Vinyals, Quoc V. Le, "Sequence to Sequence Learning with Neural Networks," arXiv:1409.3215 (2014). [`arXiv
<https://arxiv.org/abs/1409.3215>`_]

Oriol Vinyals, Quoc Le, "A Neural Conversational Model," arXiv:1506.05869 (2015). [`arXiv
<https://arxiv.org/abs/1506.05869>`_]

Tom Young, Devamanyu Hazarika, Soujanya Poria, Erik Cambria, "Recent Trends in Deep Learning Based Natural Language Processing," arXiv:1708.02709 (2017). [`arXiv
<https://arxiv.org/abs/1708.02709>`_]

Zackary C. Lipton, John Berkowitz, "A Critical Review of Recurrent Neural Networks for Sequence Learning," arXiv:1506.00019 (2015). [`arXiv
<https://arxiv.org/abs/1506.00019>`_]

