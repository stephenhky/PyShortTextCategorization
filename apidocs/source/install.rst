Installation Guide
==================

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

.. _Github: https://github.com/stephenhky/PyShortTextCategorization

Required Packages
-----------------

- Numpy_ (Numerical Python)
- SciPy_ (Scientific Python)
- Scikit-Learn_ (Machine Learning in Python)
- Theano_ (Symbolic Computing for Deep Learning)
- keras_ (Deep Learning Library for Theano and Tensorflow)
- gensim_ (Topic Modeling for Humans)
- Pandas_ (Python Data Analysis Library)
- spaCy_ (Industrial Strenglth Natural Language Processing in Python)
- stemming_ (stemming in Python)

.. _Numpy: http://www.numpy.org/
.. _SciPy: https://www.scipy.org/
.. _Scikit-Learn: http://scikit-learn.org/stable/
.. _Theano: http://deeplearning.net/software/theano/
.. _keras: https://keras.io/
.. _gensim: https://radimrehurek.com/gensim/
.. _Pandas: http://pandas.pydata.org/
.. _spaCy: https://spacy.io/
.. _stemming: https://pypi.python.org/pypi/stemming/