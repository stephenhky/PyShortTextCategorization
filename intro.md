## Introduction

This package `shorttext` is a Python package that facilitates supervised and unsupervised
learning for short text categorization. Due to the sparseness of words and
the lack of information carried in the short texts themselves, an intermediate
representation of the texts and documents are needed before they are put into
any classification algorithm. In this package, it facilitates various types
of these representations, including topic modeling and word-embedding algorithms.

Since release 1.0.0, `shorttext` runs on Python 2.7, 3.5, and 3.6.

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

[FAQ](https://shorttext.readthedocs.io/en/latest/faq.html).

## Installation

To install it, in a console, use `pip`.

```
>>> pip install -U shorttext
```

or, if you want the most updated code that is not released on PyPI yet, type

```
>>> pip install -U git+https://github.com/stephenhky/PyShortTextCategorization@master
```

Developers are advised to make sure `Keras` >=2 be installed. Users are advised to install the backend `Tensorflow` (preferred) or `Theano` in advance. It is desirable if `Cython` has been previously installed too.

Before using, check the language model of spaCy has been installed or updated, by running:

```
>>> python -m spacy download en
```

See [tutorial](http://shorttext.readthedocs.io/en/latest/tutorial.html) for how to use the package.

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
