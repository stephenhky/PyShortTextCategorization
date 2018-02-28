Character to One-Hot Vector
===========================

Since version 0.6.1, the package `shorttext` deals with character-based model. A first important
component of character-based model is to convert every character to a one-hot vector. We provide a class
:class:`shorttext.generators.SentenceToCharVecEncoder` to deal with this. Thi class incorporates
the `OneHotEncoder` in `scikit-learn` and `Dictionary` in `gensim`.

To use this, import the packages first:

>>> import numpy as np
>>> import shorttext

Then we incorporate a text file as the source of all characters to be coded. In this case, we choose
the file `big.txt` in Peter Norvig's websites:

>>> import urllib2
>>> textfile = urllib2.urlopen('http://norvig.com/big.txt', 'r')

Then instantiate the class using the function :func:`shorttext.generators.initSentenceToCharVecEncoder`:

>>> chartovec_encoder = shorttext.generators.initSentenceToCharVecEncoder(textfile)

Now, the object `chartovec_encoder` is an instance of :class:`shorttext.generators.SentenceToCharVecEncoder` . The
default signal character is `\n`, which is also encoded, and can be checked by looking at the field:

>>> chartovec_encoder.signalchar

We can convert a sentence into a bunch of one-hot vectors in terms of a matrix. For example,

>>> chartovec_encoder.encode_sentence('Maryland blue crab!', 100)
<1x93 sparse matrix of type '<type 'numpy.float64'>'
	with 1 stored elements in Compressed Sparse Column format>

This outputs a sparse matrix. Depending on your needs, you can add signal character to the beginning
or the end of the sentence in the output matrix by:

>>> chartovec_encoder.encode_sentence('Maryland blue crab!', 100, startsig=True, endsig=False)
>>> chartovec_encoder.encode_sentence('Maryland blue crab!', 100, startsig=False, endsig=True)

We can also convert a list of sentences by

>>> chartovec_encoder.encode_sentences(sentences, 100, startsig=False, endsig=True, sparse=False)

You can decide whether or not to output a sparse matrix by specifiying the parameter `sparse`.

Reference
---------

Aurelien Geron, *Hands-On Machine Learning with Scikit-Learn and TensorFlow* (Sebastopol, CA: O'Reilly Media, 2017). [`O\'Reilly
<http://shop.oreilly.com/product/0636920052289.do>`_]

Home: :doc:`index`