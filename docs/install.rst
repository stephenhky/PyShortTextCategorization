Installation
============

PIP
---

Package `shorttext` runs in Python 3.6, 3.7, and 3.8. However, for Python>=3.7, the backend
of keras_ cannot be Tensorflow_.

To install the package in Linux or OS X, enter the following in the console:

::

   pip install -U shorttext

It is very possible that you have to do it as root, that you have to add ``sudo`` in
front of the command.

On the other hand, to get the development version on Github, you can install from Github_:

::

    pip install -U git+https://github.com/stephenhky/PyShortTextCategorization@master

By adding ``-U`` in the command, it automatically installs the required packages. If not,
you have to install these packages on your own.


Backend for Keras
-----------------

The package keras_ (version >= 2.0.0) uses either Tensorflow_, Theano_, or CNTK_ as the backend, while Theano is usually
the default. However, it is highly recommended to use Tensorflow as the backend.
Users are advised to install the backend Tensorflow_ (preferred for Python 2.7, 3.5, and 3.6) or
Theano_ (preferred for Python 3.7) in advance. Refer to
:doc:`faq` for how to switch the backend. It is also desirable if the package Cython_ has been previously installed.


Possible Solutions for Installation Failures
--------------------------------------------

Most developers can install `shorttext` with the instructions above. If the installation fails,
you may try one (or more) of the following:

1. Installing Python-dev by typing:


::

    pip install -U python3-dev



2. Installing `gcc` by entering

::

    apt-get install libc6



.. _Github: https://github.com/stephenhky/PyShortTextCategorization

Required Packages
-----------------

- Numpy_ (Numerical Python, version >= 1.16.0)
- SciPy_ (Scientific Python, version >= 1.2.0)
- Scikit-Learn_ (Machine Learning in Python)
- keras_ (Deep Learning Library for Theano and Tensorflow, version >= 2.3.0)
- gensim_ (Topic Modeling for Humans, version >= 3.2.0)
- Pandas_ (Python Data Analysis Library)
- PuLP_ (Optimization with PuLP)
- snowballstemmer_ (Snowball Stemmer)
- TensorFlow_ (TensorFlow, >= 2.0.0)
- Flask_ (Flask)
- Joblib_ (Joblib: lightweight Python pipelining)

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
.. _PuLP: https://pythonhosted.org/PuLP/
.. _snowballstemmer: https://github.com/snowballstem/snowball
.. _TensorFlow: https://www.tensorflow.org/
.. _Flask: https://flask.palletsprojects.com/
.. _Joblib: https://joblib.readthedocs.io/en/latest/