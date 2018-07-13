Installation
============

PIP
---

Package `shorttext` runs in Python 2.7, 3.5, and 3.6.

To install the package in Linux or OS X, enter the following in the console:

::

   pip install -U shorttext

It is very possible that you have to do it as root, that you have to add ``sudo`` in
front of the command.

However, the repository on Python Package Index is not always the most updated. To get
the most updated (not official) version, you can install from Github_:

::

    pip install -U git+https://github.com/stephenhky/PyShortTextCategorization@master

By adding ``-U`` in the command, it automatically installs the required packages. If not,
you have to install these packages on your own.

Before using, check the language model of spaCy has been installed or updated, by running:

::

    spacy download en

The package keras_ (version >= 2.0.0) uses either Tensorflow_, Theano_, or CNTK_ as the backend, while Theano is usually
the default. However, it is highly recommended to use Tensorflow as the backend.
Users are advised to install the backend `Tensorflow` (preferred) or `Theano` in advance. Refer to
:doc:`faq` for how to switch the backend. It is also desirable if the package Cython_ has been previously installed.

.. _Github: https://github.com/stephenhky/PyShortTextCategorization

Required Packages
-----------------

- Numpy_ (Numerical Python, version >= 1.11.3)
- SciPy_ (Scientific Python, version >= 0.18.1)
- Scikit-Learn_ (Machine Learning in Python)
- keras_ (Deep Learning Library for Theano and Tensorflow, version >= 2.0.0)
- gensim_ (Topic Modeling for Humans, version >= 3.2.0)
- Pandas_ (Python Data Analysis Library)
- spaCy_ (Industrial Strenglth Natural Language Processing in Python, version >= 1.7.0)
- PuLP_ (Optimization with PuLP)
- PyStemmer_ (Snowball Stemmer, the package stemming_ is no longer used)

Home: :doc:`index`

.. _Cython: http://cython.org/
.. _Numpy: http://www.numpy.org/
.. _SciPy: https://www.scipy.org/
.. _Scikit-Learn: http://scikit-learn.org/stable/
.. _Tensorflow: https://www.tensorflow.org/
.. _Theano: http://deeplearning.net/software/theano/
.. _CNTK: https://github.com/Microsoft/CNTK/wiki
.. _keras: https://keras.io/
.. _gensim: https://radimrehurek.com/gensim/
.. _Pandas: http://pandas.pydata.org/
.. _spaCy: https://spacy.io/
.. _stemming: https://pypi.python.org/pypi/stemming/
.. _PuLP: https://pythonhosted.org/PuLP/
.. _PyStemmer: http://snowball.tartarus.org/
