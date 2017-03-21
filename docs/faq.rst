Frequently Asked Questions (FAQ)
================================

1. Can we use Tensorflow backend?

Ans: Yes, users can use tensorflow backened instead of theano backend, as both as supported
by Keras. Refer to `Keras Backend
<https://keras.io/backend/>`_ for information about switching backends.

2. Can we use word-embedding algorithms other than Word2Vec?

Ans: Currently only Word2Vec is directly supported. However, you can
convert GloVe models into Word2Vec models. See: :doc:`tutorial_wordembed` .

3. Can this package work on Python 3?

Ans: This package is written in Python 2.7. It is not guaranteed that the package works perfectly
well in Python 3.

4. This package requires SpaCy, which involves loading several models that
are needed for `shorttext` to run correctly. It gives error whenever I ran
models that require tokenization. What should I do?

If your code gives the error message that includes the following:

::

    ValueError: Found English model at //anaconda/lib/python2.7/site-packages/spacy/data/en-1.1.0.
    This model is not compatible with the current version.
    See https://spacy.io/docs/usage/models to download the new model.

Then run the following command in your terminal or console:

::

    python -m spacy download en

Refer to `<https://spacy.io/docs/usage/models`_ for more information.

Home: :doc:`index`