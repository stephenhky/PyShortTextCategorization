# Short Text Mining in Python

[![Build Status](https://travis-ci.org/stephenhky/PyShortTextCategorization.svg?branch=master)](https://travis-ci.org/stephenhky/PyShortTextCategorization)
[![GitHub release](https://img.shields.io/github/release/stephenhky/PyShortTextCategorization.svg?maxAge=3600)](https://github.com/stephenhky/PyShortTextCategorization/releases)

## Introduction

This package `shorttext` is a Python package that facilitates supervised and unsupervised
learning for short text categorization. Due to the sparseness of words and
the lack of information carried in the short texts themselves, an intermediate
representation of the texts and documents are needed before they are put into
any classification algorithm. In this package, it facilitates various types
of these representations, including topic modeling and word-embedding algorithms.

Since release 1.0.0, `shorttext` runs on Python 2.7, 3.5, and 3.6.
Since release 1.0.7, it runs on Python 3.7 as well, but the backend for `keras` cannot be `TensorFlow`.
Since release 1.0.8, it runs on Python 3.7 with 'TensorFlow' being the backend for `keras`.

Characteristics:

- example data provided (including subject keywords and NIH RePORT);
- text preprocessing;
- pre-trained word-embedding support;
- `gensim` topic models (LDA, LSI, Random Projections) and autoencoder;
- topic model representation supported for supervised learning using `scikit-learn`;
- cosine distance classification;
- neural network classification (including ConvNet, and C-LSTM);
- maximum entropy classification;
- metrics of phrases differences, including soft Jaccard score (using Damerau-Levenshtein distance), and Word Mover's distance (WMD);
- character-level sequence-to-sequence (seq2seq) learning; and
- spell correction.

## Documentation

Documentation and tutorials for `shorttext` can be found here: [http://shorttext.rtfd.io/](http://shorttext.rtfd.io/).

See [tutorial](http://shorttext.readthedocs.io/en/latest/tutorial.html) for how to use the package, and [FAQ](https://shorttext.readthedocs.io/en/latest/faq.html).

## Installation

To install it, in a console, use `pip`.

```
>>> pip install -U shorttext
```

or, if you want the most recent development version on Github, type

```
>>> pip install -U git+https://github.com/stephenhky/PyShortTextCategorization@master
```

Developers are advised to make sure `Keras` >=2 be installed. Users are advised to install the backend `Tensorflow` (preferred) or `Theano` in advance. It is desirable if `Cython` has been previously installed too.

Before using, check the language model of spaCy has been installed or updated, by running:

```
>>> python -m spacy download en
```

See [installation guide](https://shorttext.readthedocs.io/en/latest/install.html) for more details.


## Issues

To report any issues, go to the [Issues](https://github.com/stephenhky/PyShortTextCategorization/issues) tab of the Github page and start a thread.
It is welcome for developers to submit pull requests on their own
to fix any errors.

## Contributors

If you would like to contribute, feel free to submit the pull requests. You can talk to me in advance through e-mails or the
[Issues](https://github.com/stephenhky/PyShortTextCategorization/issues) page.

## Useful Links

* Documentation: [http://shorttext.readthedocs.io](http://shorttext.readthedocs.io/)
* Github: [https://github.com/stephenhky/PyShortTextCategorization](https://github.com/stephenhky/PyShortTextCategorization)
* PyPI: [https://pypi.org/project/shorttext/](https://pypi.org/project/shorttext/)
* "Package shorttext 1.0.0 released," [Medium](https://medium.com/@stephenhky/package-shorttext-1-0-0-released-ca3cb24d0ff3)
* "Python Package for Short Text Mining", [WordPress](https://datawarrior.wordpress.com/2016/12/22/python-package-for-short-text-mining/)
* "Document-Term Matrix: Text Mining in R and Python," [WordPress](https://datawarrior.wordpress.com/2018/01/22/document-term-matrix-text-mining-in-r-and-python/)
* An [earlier version](https://github.com/stephenhky/PyShortTextCategorization/tree/b298d3ce7d06a9b4e0f7d32f27bab66064ba7afa) of this repository is a demonstration of the following blog post: [Short Text Categorization using Deep Neural Networks and Word-Embedding Models](https://datawarrior.wordpress.com/2016/10/12/short-text-categorization-using-deep-neural-networks-and-word-embedding-models/)


## News

* 07/07/2019: `shorttext` 1.1.3 released.
* 06/05/2019: `shorttext` 1.1.2 released.
* 04/23/2019: `shorttext` 1.1.1 released.
* 03/03/2019: `shorttext` 1.1.0 released.
* 02/14/2019: `shorttext` 1.0.8 released.
* 01/30/2019: `shorttext` 1.0.7 released.
* 01/29/2019: `shorttext` 1.0.6 released.
* 01/13/2019: `shorttext` 1.0.5 released.
* 10/03/2018: `shorttext` 1.0.4 released.
* 08/06/2018: `shorttext` 1.0.3 released.
* 07/24/2018: `shorttext` 1.0.2 released.
* 07/17/2018: `shorttext` 1.0.1 released.
* 07/14/2018: `shorttext` 1.0.0 released.
* 06/18/2018: `shorttext` 0.7.2 released.
* 05/30/2018: `shorttext` 0.7.1 released.
* 05/17/2018: `shorttext` 0.7.0 released.
* 02/27/2018: `shorttext` 0.6.0 released.
* 01/19/2018: `shorttext` 0.5.11 released.
* 01/15/2018: `shorttext` 0.5.10 released.
* 12/14/2017: `shorttext` 0.5.9 released.
* 11/08/2017: `shorttext` 0.5.8 released.
* 10/27/2017: `shorttext` 0.5.7 released.
* 10/17/2017: `shorttext` 0.5.6 released.
* 09/28/2017: `shorttext` 0.5.5 released.
* 09/08/2017: `shorttext` 0.5.4 released.
* 09/02/2017: end of GSoC project. ([Report](https://rare-technologies.com/chinmayas-gsoc-2017-summary-integration-with-sklearn-keras-and-implementing-fasttext/))
* 08/22/2017: `shorttext` 0.5.1 released.
* 07/28/2017: `shorttext` 0.4.1 released.
* 07/26/2017: `shorttext` 0.4.0 released.
* 06/16/2017: `shorttext` 0.3.8 released.
* 06/12/2017: `shorttext` 0.3.7 released.
* 06/02/2017: `shorttext` 0.3.6 released.
* 05/30/2017: GSoC project ([Chinmaya Pancholi](https://rare-technologies.com/google-summer-of-code-2017-week-1-on-integrating-gensim-with-scikit-learn-and-keras/), with [gensim](https://radimrehurek.com/gensim/))
* 05/16/2017: `shorttext` 0.3.5 released.
* 04/27/2017: `shorttext` 0.3.4 released.
* 04/19/2017: `shorttext` 0.3.3 released.
* 03/28/2017: `shorttext` 0.3.2 released.
* 03/14/2017: `shorttext` 0.3.1 released.
* 02/23/2017: `shorttext` 0.2.1 released.
* 12/21/2016: `shorttext` 0.2.0 released.
* 11/25/2016: `shorttext` 0.1.2 released.
* 11/21/2016: `shorttext` 0.1.1 released.

## Possible Future Updates

- [ ] Using `word2vec-api` for faster loading (especially on Cloud);
- [ ] More scalability using `horovod`;
- [ ] Including BERT models;
- [ ] Use of DASK;
- [ ] Dividing components to other packages;
- [ ] More available corpus.
