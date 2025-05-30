Installation
============

PIP
---

Package `shorttext` runs in Python 3.9, 3.10, 3.11, and 3.12. However, for Python>=3.7, the backend
of keras_ cannot be Tensorflow_.

To install the package in Linux or OS X, enter the following in the console:

::

   pip install shorttext

It is very possible that you have to do it as root, that you have to add ``sudo`` in
front of the command.

On the other hand, to get the development version on Github, you can install from Github_:

::

    pip install git+https://github.com/stephenhky/PyShortTextCategorization@master


Backend for Keras
-----------------

We use TensorFlow for `keras`.

Possible Solutions for Installation Failures
--------------------------------------------

Most developers can install `shorttext` with the instructions above. If the installation fails,
you may try one (or more) of the following:

1. Installing Python-dev by typing:


::

    pip install python3-dev



2. Installing `gcc` by entering

::

    apt-get install libc6



.. _Github: https://github.com/stephenhky/PyShortTextCategorization


Home: :doc:`index`

.. _Numpy: http://www.numpy.org/
.. _SciPy: https://www.scipy.org/
.. _Scikit-Learn: http://scikit-learn.org/stable/
.. _Tensorflow: https://www.tensorflow.org/
.. _Theano: http://deeplearning.net/software/theano/
.. _CNTK: https://github.com/Microsoft/CNTK/wiki
.. _keras: https://keras.io/
.. _gensim: https://radimrehurek.com/gensim/
.. _Pandas: http://pandas.pydata.org/
.. _snowballstemmer: https://github.com/snowballstem/snowball
.. _Joblib: https://joblib.readthedocs.io/en/latest/