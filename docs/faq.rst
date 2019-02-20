Frequently Asked Questions (FAQ)
================================

**Q1. Can we use TensorFlow backend?**

Ans: Yes, users can use TensorFlow and CNTK backend instead of Theano backend. Refer to `Keras Backend
<https://keras.io/backend/>`_ for information about switching backends.


**Q2. Can we use word-embedding algorithms other than Word2Vec?**

Ans: Yes. Besides Word2Vec, you can use FastText and Poincaré embedding. See: :doc:`tutorial_wordembed` .


**Q3. Can this package work on Python 3?**

Ans: Since release 1.0.0, this package can be run in Python 2.7, 3.5, and 3.6. (Before that, it operates only under Python 2.7.)
     Since release 1.0.7, this package can also be run in Python 3.7 as well.



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

    python -m spacy download en


You might need administrator priveledge to do this. This will link a folder `en_core_web_sm` to `en` too.

You may receive this error message:

::

    Error: Couldn't link model to 'en'
    Creating a symlink in spacy/data failed. Make sure you have the required
    permissions and try re-running the command as admin, or use a
    virtualenv. You can still import the model as a module and call its
    load() method, or create the symlink manually.

This means the linkage fails. Delete the existing `en` link and do this:

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


**Q8. Is there any reasons why word-embedding keras layers no longer used since release 0.5.11?**

Ans: This functionality is removed since release 0.5.11, due to the following reasons:

* `keras` changed its code that produces this bug;
* the layer is consuming memory;
* only Word2Vec is supported; and
* the results are incorrect.


**Q9. I am having trouble in install `shorttext` on Google Cloud Platform. What should I do?**

Ans: There is no "Python.h". Run: `sudo apt-get install python3-dev` in SSH shell of the VM instance.





Home: :doc:`index`
