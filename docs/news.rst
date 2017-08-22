News
====

* 08/22/2017: `shorttext` 0.5.1 released.
* 07/28/2017: `shorttext` 0.4.1 released.
* 07/26/2017: `shorttext` 0.4.0 released.
* 06/16/2017: `shorttext` 0.3.8 released.
* 06/12/2017: `shorttext` 0.3.7 released.
* 06/02/2017: `shorttext` 0.3.6 released.
* 05/30/2017: GSoC project (`Chinmaya Pancholi
  <https://rare-technologies.com/google-summer-of-code-2017-week-1-on-integrating-gensim-with-scikit-learn-and-keras/>`_ ).
* 05/16/2017: `shorttext` 0.3.5 released.
* 04/27/2017: `shorttext` 0.3.4 released.
* 04/19/2017: `shorttext` 0.3.3 released.
* 03/28/2017: `shorttext` 0.3.2 released.
* 03/14/2017: `shorttext` 0.3.1 released.
* 02/23/2017: `shorttext` 0.2.1 released.
* 12/21/2016: `shorttext` 0.2.0 released.
* 11/25/2016: `shorttext` 0.1.2 released.
* 11/21/2016: `shorttext` 0.1.1 released.

What's New
----------

Release 0.5.1 (August 22, 2017)
-------------------------------

* Implementation of Damerau-Levenshtein distance and soft Jaccard score;
* Implementation of Word Mover's distance.


Release 0.4.1 (July 28, 2017)
-----------------------------

* Further Travis.CI update tests;
* Model file I/O updated (for huge models);
* Migrating documentation to [readthedocs.org](readthedocs.org); previous documentation at `Pythonhosted.org` destroyed.


Release 0.4.0 (July 26, 2017)
-----------------------------

* Maximum entropy models;
* Use of `gensim` Word2Vec `keras` layers;
* Incorporating new features from `gensim`;
* Use of Travis.CI for pull request testing.

Release 0.3.8 (June 16, 2017)
-----------------------------

* Bug fixed on `sumvecframeworks`.

Release 0.3.7 (June 12, 2017)
-----------------------------

* Bug fixed on `VarNNSumEmbedVecClassifier`.

Release 0.3.6 (June 2, 2017)
----------------------------

* Added deprecation decorator;
* Fixed path configurations;
* Added "update" corpus capability to `gensim` models.

Google Summer of Code (May 30, 2017)
-----------------------------------------

Chinamaya Pancholi, a Google Summer of Code (GSoC) student, is involved in
the open-source development of `gensim`, that his project will be very related
to the `shorttext` package. More information can be found in his first `blog entry
<https://rare-technologies.com/google-summer-of-code-2017-week-1-on-integrating-gensim-with-scikit-learn-and-keras/>`_ .

Release 0.3.5 (May 16, 2017)
----------------------------

* Refactoring topic modeling to generators subpackage, but keeping package backward compatible.
* Added Inaugural Addresses as an example training data;
* Fixed bugs about package paths.

Release 0.3.4 (Apr 27, 2017)
----------------------------

* Fixed relative path loading problems.

Release 0.3.3 (Apr 19, 2017)
----------------------------

* Deleted `CNNEmbedVecClassifier`;
* Added script `ShortTextWord2VecSimilarity`.

`More Info
<https://datawarrior.wordpress.com/2017/04/20/release-of-shorttext-0-3-3/>`_


Release 0.3.2 (Mar 28, 2017)
----------------------------

* Bug fixed for `gensim` model I/O;
* Console scripts update;
* Neural networks up to Keras 2 standard (refer to `this
  <https://github.com/fchollet/keras/wiki/Keras-2.0-release-notes/>`_ ).

Release 0.3.1 (Mar 14, 2017)
----------------------------

* Compact model I/O: all models are in single files;
* Implementation of stacked generalization using logistic regression.

Release 0.2.1 (Feb 23, 2017)
----------------------------

* Removal attempts of loading GloVe model, as it can be run using `gensim` script;
* Confirmed compatibility of the package with `tensorflow`;
* Use of `spacy` for tokenization, instead of `nltk`;
* Use of `stemming` for Porter stemmer, instead of `nltk`;
* Removal of `nltk` dependencies;
* Simplifying the directory and module structures;
* Module packages updated.

`More Info
<https://datawarrior.wordpress.com/2017/02/24/release-of-shorttext-0-2-1/>`_

Release 0.2.0 (Dec 21, 2016)
----------------------------

Home: :doc:`index`