Frequently Asked Questions (FAQ)
================================

**Q1. Can we use TensorFlow backend?**

Ans: Yes, users can use TensorFlow and CNTK backend instead of Theano backend. Refer to `Keras Backend
<https://keras.io/backend/>`_ for information about switching backends.


**Q2. Can we use word-embedding algorithms other than Word2Vec?**

Ans: Yes. Besides Word2Vec, you can use FastText and Poincaré embedding. See: :doc:`tutorial_wordembed` .


**Q3. Can this package work on Python 3?**

Ans: This package is written in Python 2.7. It is not guaranteed that the package works perfectly
well in Python 3.


**Q4. This package requires SpaCy, which involves loading several models that
are needed for `shorttext` to run correctly. It gives error whenever I ran
models that require tokenization. What should I do?**

Ans: If your code gives the error message that includes the following:

::

    ValueError: Found English model at //anaconda/lib/python2.7/site-packages/spacy/data/en-1.1.0.
    This model is not compatible with the current version.
    See https://spacy.io/docs/usage/models to download the new model.

Then run the following command in your terminal or console:

::

    spacy download en


You might need administrator priveledge to do this. This will link a folder `en_core_web_sm` to `en` too.
In case the linkage fails, delete the existing `en` link and do this:

::

    python -m spacy link en_core_web_sm en


Refer to `spaCy webpage
<https://spacy.io/docs/usage/models>`_ for more information.


**Q5. Warning or messages pop up when running models involving neural networks. What is the problem?**

Ans: Make sure your `keras` have version >= 2.


**Q6. The following error message appears while loading shorttext:**

::

    ImportError: dlopen: cannot load any more object with static TLS

**How do I deal with it?**

Ans: If you use Tensorflow as your backend, you may experience this problem. This has been pointed
out by Yeung in the community: `issue
<https://github.com/stephenhky/PyShortTextCategorization/issues/3>`_ . You should either reload tensorflow,
or reinstall, or try to workaround by importing `spaCy` before `shorttext`.


**Q7. How should I cite `shorttext` if I use it in my research?**

Ans: For the time being, You do not have to cite a particular paper for using this package.
However, if you use any particular functions or class, check out the docstring. If there is a paper (or papers)
mentioned, cite those papers. For example, if you use `CNNWordEmbed` in `frameworks
<https://github.com/stephenhky/PyShortTextCategorization/blob/master/shorttext/classifiers/embed/nnlib/frameworks.py>`_,
according to the docstring, cite Yoon Kim's paper. Refer to this documentation for the reference too.


Home: :doc:`index`
